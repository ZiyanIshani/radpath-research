"""
train_adapter.py — Train rad/path/multimodal adapters with Cox proportional hazards loss.

Architecture (corrected dims from actual encoder outputs):
  Pathology Adapter:  Linear(768, d) → ReLU → Linear(d, 512)     [pgp slide enc: 768-d]
  Radiology Adapter:  Linear(3072, d) → ReLU → Linear(d, 512)    [rad-dino × 4: 3072-d]
  Multi-Modal Adapter: Linear(1024, d) → ReLU → Linear(d, 1)     [512+512 → hazard scalar]

Loss: Cox partial log-likelihood (Breslow approximation for ties).
Metric: Concordance index (C-index).

Usage:
    python scripts/train_adapter.py \\
        --manifest data/manifest.csv \\
        --mode multimodal \\
        --d 512 \\
        --epochs 100 \\
        --lr 1e-4 \\
        --out models/checkpoints/multimodal_v1.pt

    --mode: radiology | pathology | multimodal
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Corrected embedding dimensions (from actual encoders, not CLAUDE.md table)
# ---------------------------------------------------------------------------
PATH_DIM  = 768    # gigapath_slide_enc12l768d CLS output
RAD_DIM   = 3072   # rad-dino 768-d × 4 modalities
FUSED_DIM = 1024   # 512 (path) + 512 (rad)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SurvivalDataset(Dataset):
    """
    Loads pre-computed embeddings referenced in manifest.csv.

    manifest.csv columns:
        subject_id, pgp_path, mi2_path, grade, survival_days, censored
    """

    def __init__(self, manifest: pd.DataFrame, mode: str):
        self.rows = manifest.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]

        item = {
            "survival_days": torch.tensor(row["survival_days"], dtype=torch.float32),
            "censored":      torch.tensor(row["censored"],      dtype=torch.float32),
            "grade":         torch.tensor(row["grade"],         dtype=torch.long),
        }

        if self.mode in ("pathology", "multimodal"):
            pt = torch.load(row["pgp_path"], weights_only=True)
            emb = pt["slide_embedding"] if isinstance(pt, dict) else pt
            item["path_emb"] = emb.float()  # (768,)

        if self.mode in ("radiology", "multimodal"):
            pt = torch.load(row["mi2_path"], weights_only=True)
            emb = pt["mi2_embedding"] if isinstance(pt, dict) else pt
            item["rad_emb"] = emb.float()   # (3072,)

        return item


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PathologyAdapter(nn.Module):
    def __init__(self, d: int = 512):
        super().__init__()
        self.enc = MLP(PATH_DIM, d, 512)

    def forward(self, x):
        return self.enc(x)  # (B, 512)


class RadiologyAdapter(nn.Module):
    def __init__(self, d: int = 512):
        super().__init__()
        self.enc = MLP(RAD_DIM, d, 512)

    def forward(self, x):
        return self.enc(x)  # (B, 512)


class MultiModalAdapter(nn.Module):
    def __init__(self, d: int = 512):
        super().__init__()
        self.hazard = MLP(FUSED_DIM, d, 1)

    def forward(self, path_out, rad_out):
        fused = torch.cat([path_out, rad_out], dim=-1)  # (B, 1024)
        h = self.hazard(fused).squeeze(-1)              # (B,)
        return torch.clamp(h, -3.0, 3.0)


class SurvivalModel(nn.Module):
    """Wraps adapters for a given mode."""

    def __init__(self, mode: str, d: int = 512):
        super().__init__()
        self.mode = mode
        if mode in ("pathology", "multimodal"):
            self.path_adapter = PathologyAdapter(d)
        if mode in ("radiology", "multimodal"):
            self.rad_adapter = RadiologyAdapter(d)
        if mode == "multimodal":
            self.mm_adapter = MultiModalAdapter(d)
        elif mode == "pathology":
            # Solo pathology: path_out → hazard
            self.hazard_head = nn.Sequential(nn.Linear(512, d), nn.ReLU(), nn.Linear(d, 1))
        elif mode == "radiology":
            # Solo radiology: rad_out → hazard
            self.hazard_head = nn.Sequential(nn.Linear(512, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, batch):
        if self.mode == "pathology":
            out = self.path_adapter(batch["path_emb"])
            h = self.hazard_head(out).squeeze(-1)
            return torch.clamp(h, -3.0, 3.0)

        if self.mode == "radiology":
            out = self.rad_adapter(batch["rad_emb"])
            h = self.hazard_head(out).squeeze(-1)
            return torch.clamp(h, -3.0, 3.0)

        # multimodal
        p = self.path_adapter(batch["path_emb"])
        r = self.rad_adapter(batch["rad_emb"])
        return self.mm_adapter(p, r)


# ---------------------------------------------------------------------------
# Cox loss (Breslow approximation)
# ---------------------------------------------------------------------------

def cox_loss(hazard: torch.Tensor, survival_days: torch.Tensor, censored: torch.Tensor) -> torch.Tensor:
    """
    Negative Cox partial log-likelihood (Breslow approximation).

    Args:
        hazard:        (B,) predicted log-hazard values
        survival_days: (B,) observed times (larger = longer survival)
        censored:      (B,) 1 if censored (no event), 0 if event observed

    Returns:
        scalar loss
    """
    # Sort descending by time (longest survivors first)
    order = torch.argsort(survival_days, descending=True)
    h = hazard[order]
    e = 1.0 - censored[order]   # event indicator: 1 = event occurred

    # Log-sum-exp over risk set (cumulative from longest to shortest time)
    log_risk = torch.logcumsumexp(h, dim=0)

    # Partial log-likelihood: sum over events only
    uncensored_ll = (h - log_risk) * e
    n_events = e.sum().clamp(min=1)
    return -uncensored_ll.sum() / n_events


# ---------------------------------------------------------------------------
# C-index
# ---------------------------------------------------------------------------

def concordance_index(hazard: np.ndarray, survival_days: np.ndarray, censored: np.ndarray) -> float:
    """
    Harrell's C-index.
    concordant pair: higher hazard → shorter survival (event occurred).
    """
    n = len(hazard)
    concordant = 0
    permissible = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if censored[i] == 1:   # i is censored, can't use as reference
                continue
            if survival_days[i] < survival_days[j]:
                permissible += 1
                if hazard[i] > hazard[j]:
                    concordant += 1
                elif hazard[i] == hazard[j]:
                    concordant += 0.5
    return concordant / max(permissible, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Mode: {args.mode}  |  d={args.d}")

    manifest = pd.read_csv(args.manifest)
    print(f"Subjects: {len(manifest)}")

    # Simple 80/20 train/val split (no shuffle for reproducibility)
    n_val = max(1, int(0.2 * len(manifest)))
    val_df   = manifest.iloc[-n_val:].copy()
    train_df = manifest.iloc[:-n_val].copy()

    train_ds = SurvivalDataset(train_df, args.mode)
    val_ds   = SurvivalDataset(val_df,   args.mode)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SurvivalModel(args.mode, d=args.d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    best_cindex = -1.0
    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            hazard = model(batch)
            loss = cox_loss(hazard, batch["survival_days"], batch["censored"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # --- Validate ---
        model.eval()
        all_h, all_t, all_c = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                h = model(batch)
                all_h.append(h.cpu().numpy())
                all_t.append(batch["survival_days"].cpu().numpy())
                all_c.append(batch["censored"].cpu().numpy())

        h_arr = np.concatenate(all_h)
        t_arr = np.concatenate(all_t)
        c_arr = np.concatenate(all_c)
        cindex = concordance_index(h_arr, t_arr, c_arr)

        avg_loss = train_loss / max(len(train_loader), 1)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}  val_cindex={cindex:.4f}")

        if cindex > best_cindex:
            best_cindex = cindex
            torch.save({
                "epoch":    epoch,
                "mode":     args.mode,
                "d":        args.d,
                "cindex":   cindex,
                "model_state": model.state_dict(),
                "path_dim": PATH_DIM,
                "rad_dim":  RAD_DIM,
            }, args.out)

    print(f"\nBest val C-index: {best_cindex:.4f}  →  saved to {args.out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train survival adapters with Cox loss.")
    ap.add_argument("--manifest",   required=True,          help="Path to manifest.csv")
    ap.add_argument("--mode",       default="multimodal",   choices=["radiology", "pathology", "multimodal"])
    ap.add_argument("--d",          type=int, default=512,  help="Adapter hidden dim")
    ap.add_argument("--epochs",     type=int, default=100)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out",        required=True,          help="Checkpoint output path (.pt)")
    train(ap.parse_args())
