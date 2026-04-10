import os
import argparse
import random
import time
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as opt
from dataset import FullDataset
from LoRA_SAM3 import LoRA_SAM3
from helpers.benchmark import print_trainable_params
from helpers.save_model import get_trainable_state_dict
from helpers.load_sam3 import load_sam3, create_vit_backbone, load_sam3_checkpoint_to_504
from helpers.loss import structure_loss
import wandb

# -----------------------------
# Seeding
# -----------------------------
def seed_torch(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# -----------------------------
# Training
# -----------------------------
def main(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch()

    # WandB handling for resume
    wandb_id = wandb.util.generate_id()
    wandb.init(
        project="LoRA_SAM3_MAS_Ablations",     
        name="LoRA-SAM3",   
        id=wandb_id,
        resume="allow",
        config={
            "epochs": args.epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "dataset": args.path,
        }
    )

    # Dataset
    train_dataset = FullDataset(args, args.size, mode='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.num_workers * 2,
    )

    os.makedirs(args.save_path, exist_ok=True)

    # -----------------------------
    # 1. Build Model & Base Setup
    # -----------------------------
    # Load SAM3 encoder
    sam3_vit = create_vit_backbone(args.size)
    sam3_vit = load_sam3_checkpoint_to_504(sam3_vit, args.sam_ckpt)
    
    sam3 = load_sam3(args.sam_ckpt, device)
    sam3.backbone.vision_backbone.trunk = sam3_vit

    # Build model
    model = LoRA_SAM3(sam3)
    model.to(device)
    
    # -----------------------------
    # 2. Setup Optimizer & Scheduler
    # -----------------------------
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = opt.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_iters = len(train_loader) * args.epoch
    
    # We initialize scheduler here, but we might update 'last_epoch' later if resuming
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_iters,
        eta_min=1e-7,
    )

    # -----------------------------
    # 3. Load Checkpoint (Resume or Transfer)
    # -----------------------------
    start_epoch = 0
    best_loss = float('inf')

    # Consolidate loading logic to handle both --checkpoint and --resume
    load_path = args.resume if (args.resume and os.path.isfile(args.resume)) else args.checkpoint

    if load_path and os.path.isfile(load_path):
        print(f"Loading checkpoint from {load_path}")
        ckpt = torch.load(load_path, map_location=device)
        
        # --- 1. EXTRACT WEIGHTS ---
        if 'model_state_dict' in ckpt:
            weights = ckpt['model_state_dict']
        else:
            weights = ckpt

        # --- 2. RESTORE METADATA (Resume Only) ---
        if args.resume:
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    print("Optimizer state restored.")
                except Exception as e:
                    print(f"Warning: Optimizer mismatch, skipping. {e}")
            
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch'] + 1
                print(f"Resuming from Epoch {start_epoch}")
                
            if 'best_loss' in ckpt:
                best_loss = ckpt['best_loss']

        # --- 3. LOAD BACKBONE ---
        new_ckpt = dict()
        for k, v in weights.items():
             if "detector.backbone.vision_backbone.trunk" in k and 'freqs_cis' not in k:
                 new_ckpt[k[len("detector.backbone.vision_backbone.trunk."):]] = v
        
        if len(new_ckpt) > 0:
            sam3_vit.load_state_dict(new_ckpt, strict=False)
        
        # --- 4. LOAD MODEL ---
        model.load_state_dict(weights, strict=False)
        
    # --- SCHEDULER FIX ---
    # Fast-forward scheduler if we resumed mid-training
    if start_epoch > 0:
        if scheduler.last_epoch == 0:
            print(f"Fast-forwarding scheduler to Epoch {start_epoch}")
            steps_to_skip = start_epoch * total_iters
            scheduler.last_epoch = steps_to_skip
            scheduler.step() 


    # -----------------------------
    # 4. Epoch Loop
    # -----------------------------
    print_trainable_params(model)
    model.train()
    
    print(f"Starting training from epoch {start_epoch + 1} to {args.epoch}")

    for epoch in range(start_epoch, args.epoch):
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            rgb = batch['image'].to(device)
            target = batch['label'].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred0, pred1 = model(rgb)
                loss0 = structure_loss(pred0, target) 
                loss1 = structure_loss(pred1, target)
                batch_loss = loss0 + loss1
                
                # Nan check
                if torch.isnan(batch_loss):
                    print(f"Loss NaN at epoch {epoch}, iter {i}. Skipping.")
                    continue
                    
                epoch_loss += batch_loss.item()

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            scheduler.step()

            if i % 50 == 0:
                print("epoch:{}-{}: loss:{:.4f} lr:{:.8f}".format(
                    epoch + 1, i + 1, batch_loss.item(), optimizer.param_groups[0]["lr"]))


        epoch_loss /= len(train_loader)    

        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # -----------------------------
        # Save Logic (Updated to Hybrid Format)
        # -----------------------------
        # 1. Get weights
        checkpoint_dict = get_trainable_state_dict(model)
        
        # 2. Inject metadata
        checkpoint_dict['epoch'] = epoch
        checkpoint_dict['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()
        checkpoint_dict['best_loss'] = best_loss

        if (epoch + 1) % 5 == 0 or epoch == args.epoch - 1:
            checkpoint_path = os.path.join(args.save_path, f'LoRA-SAM3-{epoch + 1}.pth')
            torch.save(checkpoint_dict, checkpoint_path)
            print(f"[Checkpoint Saved at {checkpoint_path}]")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Update the best_loss value in the dict before saving
            checkpoint_dict['best_loss'] = best_loss 
            best_checkpoint_path = os.path.join(args.save_path, 'LoRA-SAM3-best.pth')
            torch.save(checkpoint_dict, best_checkpoint_path)
            print(f"[Best Checkpoint Saved at {best_checkpoint_path}]")

        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed:.2f} seconds")

    wandb.finish()

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAM3-DUNet")
    parser.add_argument("--sam_ckpt", type=str, help="Path to SAM3 checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (Weights only)")
    parser.add_argument("--resume", type=str, help="Path to resume checkpoint (Full state)")
    parser.add_argument("--path", type=str, default="../datasets/MAS/MAS3K_RMAS/")
    parser.add_argument("--save_path", type=str, default="./checkpoints/")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--size", type=int, default=672)
    args = parser.parse_args()


    ## TODO: Refactor to use config files for cleaner task/dataset management
    ######### TASK SELECTION ###########
    ## SOD ##
    # args.path = "/mnt/c/Projects/datasets/RGB/DUTS_HRSOD_UHRSD/"

    ## RGB-D SOD ##
    # args.path = "/mnt/c/Projects/datasets/RGB-D_SOD/NJU2K_NLPR/NJU2K_NLPR_Train/"

    ## COD ##
    args.path = "/mnt/c/Projects/datasets/COD/TrainDataset/"

    ## MAS ##
    # args.path = "/mnt/c/Projects/datasets/MAS3K_RMAS_COD/"
    
    ####################################
    
    args.sam_ckpt = "./sam3-main/sam3.pt"
    
    # Use --resume for continuing training, --checkpoint for loading weights only
    # args.resume = "./checkpoints/SAM3-best.pth" 
    
    args.size = 672
    args.save_path = "./checkpoints/"
    args.epoch = 20
    args.lr = 5e-4 
    args.batch_size = 8
    args.weight_decay = 5e-4
    args.num_workers = 2

    main(args)