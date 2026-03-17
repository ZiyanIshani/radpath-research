"""
mi2_encode.py — Radiology encoder using microsoft/rad-dino as MI2 substitute.

Substitution rationale:
  - MedImageInsight (MI2) is not publicly available on HuggingFace.
  - microsoft/rad-dino (DINOv2 ViT-B/14 trained on radiology images) is used instead.
  - Output dim: 768-d per modality (vs. 1024-d from MI2).
  - Concatenated 4-modality embedding: (3072,) instead of (4096,).
  - Update Radiology Adapter input: Linear(3072, d) in train_adapter.py.

Usage (in .venv on GPU node):
    python scripts/mi2_encode.py \
        --subject TCGA-06-0001 \
        --t1    data/raw/mri/TCGA-06-0001/T1.nii.gz \
        --t1pc  data/raw/mri/TCGA-06-0001/T1PC.nii.gz \
        --t2    data/raw/mri/TCGA-06-0001/T2.nii.gz \
        --flair data/raw/mri/TCGA-06-0001/FLAIR.nii.gz \
        --out   data/embeddings/TCGA-06-0001_mi2.pt
"""

import argparse
import os

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = "microsoft/rad-dino"
EMB_DIM = 768          # rad-dino CLS token dim
N_MODALITIES = 4       # T1, T1PC, T2, FLAIR
CONCAT_DIM = EMB_DIM * N_MODALITIES  # 3072


def load_nifti_center_slice(path: str) -> np.ndarray:
    """
    Load a NIfTI volume and return the center axial slice as uint8 [0,255].

    Preprocessing (following Chen et al. TMI 2022):
      - Clip intensity to [1st, 99th] percentile.
      - Normalize to [0, 255].
    """
    img = nib.load(path)
    vol = img.get_fdata(dtype=np.float32)

    # Axial slices are along the last axis (RAS orientation assumed).
    # If not canonical, reorient — for simplicity take along largest dim.
    z_dim = np.argmax(vol.shape)
    z_center = vol.shape[z_dim] // 2

    # Extract center slice
    slc = np.take(vol, z_center, axis=z_dim)  # (H, W)

    # Percentile clipping
    p1, p99 = np.percentile(slc, 1), np.percentile(slc, 99)
    slc = np.clip(slc, p1, p99)

    # Normalize to [0, 255]
    denom = p99 - p1
    if denom > 0:
        slc = (slc - p1) / denom
    slc = (slc * 255).astype(np.uint8)

    return slc  # (H, W)


def slice_to_pil(slc: np.ndarray) -> Image.Image:
    """Convert grayscale (H, W) uint8 array to 3-channel PIL Image."""
    gray = Image.fromarray(slc)  # uint8 2D array → 'L' mode inferred
    return gray.convert("RGB")


@torch.no_grad()
def encode_modalities(
    paths: dict[str, str],
    processor,
    model,
    device: str,
) -> dict[str, torch.Tensor]:
    """
    Encode each modality independently.

    Returns dict: modality_name → (768,) tensor on CPU.
    """
    results = {}
    for name, path in paths.items():
        slc = load_nifti_center_slice(path)
        img = slice_to_pil(slc)
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # CLS token: shape (1, 768)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        results[name] = cls_emb.squeeze(0).cpu()  # (768,)
    return results


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Load model
    print(f"Loading {MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    modality_paths = {
        "T1":    args.t1,
        "T1PC":  args.t1pc,
        "T2":    args.t2,
        "FLAIR": args.flair,
    }

    # Verify all paths exist
    for name, path in modality_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} volume not found: {path}")

    print("Encoding modalities ...")
    embs = encode_modalities(modality_paths, processor, model, device)
    for name, e in embs.items():
        print(f"  {name}: {tuple(e.shape)}")

    # Concatenate in canonical order: T1, T1PC, T2, FLAIR → (3072,)
    order = ["T1", "T1PC", "T2", "FLAIR"]
    concat = torch.cat([embs[m] for m in order], dim=0)  # (3072,)
    print(f"Concatenated radiology embedding: {tuple(concat.shape)}")

    torch.save({
        "subject_id":      args.subject,
        "mi2_embedding":   concat,          # (3072,) — main output
        "modality_embs":   embs,            # dict of (768,) per modality
        "model_id":        MODEL_ID,
        "emb_dim":         EMB_DIM,
        "concat_dim":      CONCAT_DIM,
        "modality_order":  order,
    }, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Encode MRI modalities with rad-dino.")
    ap.add_argument("--subject", required=True, help="Subject ID (e.g. TCGA-06-0001)")
    ap.add_argument("--t1",    required=True, help="Path to T1 NIfTI volume")
    ap.add_argument("--t1pc",  required=True, help="Path to T1-PostContrast NIfTI volume")
    ap.add_argument("--t2",    required=True, help="Path to T2 NIfTI volume")
    ap.add_argument("--flair", required=True, help="Path to T2-FLAIR NIfTI volume")
    ap.add_argument("--out",   required=True, help="Output .pt path")
    main(ap.parse_args())
