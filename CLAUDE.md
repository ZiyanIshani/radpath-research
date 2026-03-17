# CLAUDE.md — Rad-Path Survival Prediction

## Project Overview

Reimplementation of the Microsoft Azure AI Foundry rad-path survival pipeline
([blog post](https://techcommunity.microsoft.com/blog/healthcareandlifesciencesblog/cancer-survival-with-radiology-pathology-analysis-and-healthcare-ai-models-in-az/4366241))
using **locally-run** foundation models on Phoenix HPC. The goal is to predict cancer
grade and patient survival from fused radiology (MRI) + histopathology (H&E WSI) embeddings
using the TCGA-GBMLGG glioma dataset (170 subjects, grades 0–2).

Reference paper inspirations: Chen et al. 2020, Can et al. 2022.

---

## Architecture Summary

```
Radiology MRI (T1, T1-PC, T2, T2-FLAIR) ─► MedImageInsight (MI2) ─► 4x 1024-d ─► concat 4096-d
                                                                                         │
                                                                                   Radiology Adapter
                                                                                   (4096 → d, MLP)
                                                                                         │
                                                                               Output Embeddings (512-d)
                                                                                         │
                                                                              ┌──────────┴──────────┐
Histopath WSI ─► tile 256×256 @ 0.5mpp ─► Prov-GigaPath tile encoder        │    Multi-Modal       │
                    (N tiles, 1536-d each) ─► slide encoder (adaptive avg    │    Adapter (MLP)     │
                    pool) ─► slide embedding (1536-d)                        │                      │
                                                                   Pathology └──────────┬──────────┘
                                                              Adapter (1536 → d)        │
                                                              Concat embeddings    Hazard Value
                                                              (1024-d)            (range: -3 to +3)
                                                                                        │
                                                                              ┌─────────┴─────────┐
                                                                         Cancer Grade        Survival Model
                                                                         Prediction          (Cox loss, C-index)
```

**Loss function:** Cox proportional hazards loss  
**Output:** Hazard value in [-3, +3] (negative = low risk, positive = high risk)  
**Evaluation:** C-index, Kaplan-Meier curves per grade

---

## Compute Environment

### Cluster: Phoenix (Georgia Tech PACE)

```bash
# Request GPU node
srun --partition=gpu-rtx6000 --gres=gpu:1 --pty bash

# Verify GPU in .venv
source ~/radpath/.venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"  # must print True
```

### Two Environments (do NOT mix them up)

| Env | Purpose | Activate |
|-----|---------|----------|
| `conda gigapath_wsi` | WSI tiling via OpenSlide (CPU, ndpi support) | `conda activate gigapath_wsi` |
| `~/radpath/.venv` | GPU embedding — GigaPath tile/slide encoder, MI2, training | `source ~/radpath/.venv/bin/activate` |

**Rule:** Any script that calls `torch`, `timm`, or the GigaPath model must run in `.venv` on a GPU node.  
**Rule:** Any script that calls `openslide` for tiling must run in `gigapath_wsi` (CPU node is fine).

---

## Repository Layout

```
~/scratch/radpath/
├── data/
│   ├── raw/
│   │   ├── wsi/                  # .ndpi whole-slide images
│   │   └── mri/                  # per-subject MRI volumes (T1, T1PC, T2, FLAIR)
│   ├── tiles/
│   │   └── <slide_id>/           # 256×256 PNG tiles, named <x>x_<y>y.png
│   ├── embeddings/
│   │   ├── <slide_id>_pgp_slide.pt   # slide-level (1536,) tensor
│   │   └── <subject_id>_mi2.pt       # radiology (4096,) tensor
│   └── manifest.csv              # subject_id, slide_path, t1_path, t1pc_path, t2_path, flair_path, grade, survival_days, censored
├── external/
│   └── prov-gigapath/            # cloned GigaPath repo (patched — see Known Issues)
├── scripts/
│   ├── tile_wsi.py               # (gigapath_wsi env) tiles a WSI at 0.5 mpp
│   ├── pgp_encode_from_tiles.py  # (.venv + GPU) tile encoder → slide encoder → save .pt
│   ├── mi2_encode.py             # (.venv + GPU) MI2 per-modality → concat → save .pt
│   ├── build_manifest.py         # scans data dirs, writes manifest.csv
│   ├── train_adapter.py          # trains rad/path/multimodal adapters with Cox loss
│   └── evaluate.py               # C-index, KM curves from saved hazard scores
├── models/
│   └── checkpoints/              # saved adapter weights
├── notebooks/
│   └── explore.ipynb
└── CLAUDE.md                     # ← this file
```

---

## Pipeline — Step by Step

### Step 1: Tile WSIs  *(gigapath_wsi env)*

```bash
conda activate gigapath_wsi
python scripts/tile_wsi.py \
    --wsi data/raw/wsi/PROV-000-000001.ndpi \
    --out data/tiles/ \
    --target-mpp 0.5 \
    --tile-size 256
```

- Uses OpenSlide to find the level closest to 0.5 mpp.
- Saves tiles as `<x>x_<y>y.png` under `data/tiles/<slide_stem>/`.
- Expected: ~1000–2000 tiles per slide.

### Step 2: Encode Tiles → Slide Embedding  *(.venv + GPU)*

```bash
source ~/radpath/.venv/bin/activate
python scripts/pgp_encode_from_tiles.py \
    --tile-dir data/tiles/PROV-000-000001.ndpi \
    --out data/embeddings/PROV-000-000001_pgp_slide.pt
```

- Tile encoder: `gigapath_tile_encoder` → `(N, 1536)` on GPU.
- Parses `(x, y)` coords from filenames for positional input to slide encoder.
- Slide encoder: adaptive average pooling → `(1, 1536)` → squeeze → `(1536,)`.
- Saves as a `torch.Tensor` `.pt` file.

### Step 3: Encode MRI → Radiology Embedding  *(.venv + GPU)*

```bash
python scripts/mi2_encode.py \
    --subject TCGA-06-0001 \
    --t1    data/raw/mri/TCGA-06-0001/T1.nii.gz \
    --t1pc  data/raw/mri/TCGA-06-0001/T1PC.nii.gz \
    --t2    data/raw/mri/TCGA-06-0001/T2.nii.gz \
    --flair data/raw/mri/TCGA-06-0001/FLAIR.nii.gz \
    --out   data/embeddings/TCGA-06-0001_mi2.pt
```

- Preprocesses each volume to 512×512 axial slices (center crop, normalize per Chen et al. TMI 2022).
- Runs MI2 inference per modality → `(1, 1024)` each.
- Concatenates 4 modalities → `(1, 4096)` → saved as `(4096,)` tensor.
- **Status:** Not yet implemented. MI2 weights / HF model not yet pulled.

### Step 4: Build Manifest

```bash
python scripts/build_manifest.py \
    --embedding-dir data/embeddings/ \
    --clinical      data/raw/clinical.csv \
    --out           data/manifest.csv
```

`manifest.csv` schema:
```
subject_id, pgp_path, mi2_path, grade, survival_days, censored
```

### Step 5: Train Adapters  *(.venv + GPU)*

```bash
python scripts/train_adapter.py \
    --manifest data/manifest.csv \
    --mode multimodal \
    --d 512 \
    --epochs 100 \
    --lr 1e-4 \
    --out models/checkpoints/multimodal_v1.pt
```

`--mode` options: `radiology` | `pathology` | `multimodal`

**Architecture:**
- Radiology Adapter: `Linear(4096, d) + ReLU + Linear(d, 512)`
- Pathology Adapter: `Linear(1536, d) + ReLU + Linear(d, 512)`
- Multi-Modal Adapter: `Linear(1024, d) + ReLU + Linear(d, 1)` → scalar hazard

**Loss:** Negative partial log-likelihood (Cox loss). Ties handled via Breslow approximation.

### Step 6: Evaluate

```bash
python scripts/evaluate.py \
    --manifest data/manifest.csv \
    --checkpoint models/checkpoints/multimodal_v1.pt \
    --out results/
```

Outputs: `c_index.txt`, `km_curves.png` (one curve per grade).

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| GPU environment | ✅ Done | `.venv` on `gpu-rtx6000`, CUDA confirmed |
| WSI tiling | ✅ Done | 1070 tiles for sample slide `PROV-000-000001.ndpi` |
| PGP tile encoder | ✅ Done | Outputs `(1, 1536)` on GPU |
| PGP slide encoder | ✅ Done | `PROV-000-000001_pgp_slide.pt` saved; slide_embedding `(768,)` (see Note 4) |
| MI2 radiology encoder | ✅ Done | `scripts/mi2_encode.py` uses rad-dino substitute; outputs (3072,) — see Note 6 |
| Manifest builder | ❌ Not started | Blocked on having real MRI data + embeddings |
| Adapter training | ❌ Not started | Blocked on embeddings |
| Evaluation | ❌ Not started | — |

**Immediate next action:** Implement `scripts/train_adapter.py` with Cox loss. Use corrected dims: path adapter in=768, rad adapter in=3072, fused=1024.

---

## Known Issues & Patches

### 1. GigaPath slide encoder `np` NameError
**File:** `external/prov-gigapath/gigapath/torchscale/architecture/config.py`  
**Fix:** Add `import numpy as np` at the top of the file.  
**Root cause:** The config uses `eval()` to parse segment lengths containing `np.*` expressions, but numpy was not imported in that module scope.  
**Status:** ✅ Patched

### 2. OpenSlide not available in `.venv`
**Do not** attempt to install OpenSlide into `.venv`. Keep tiling strictly in `conda gigapath_wsi`.  
Mixing the two breaks GigaPath's `timm` dependency chain.

### 3. `pgp_encode_from_tiles.py` — missing batch dimension
**File:** `scripts/pgp_encode_from_tiles.py` line 67
**Fix:** Call `tile_embed.unsqueeze(0)` and `coords.unsqueeze(0)` before passing to slide encoder; unpack the returned list: `slide_out[0]`.
**Root cause:** Slide encoder `forward` expects `(1, N, D)` batch-first tensors; script passed `(N, D)`.
**Status:** ✅ Patched

### 4. Flash attention unavailable on RTX 6000 (sm_75)
**File:** `external/prov-gigapath/gigapath/torchscale/component/flash_attention.py`
**Fix:** Added standard PyTorch attention fallback (`torch.bmm`-based) in the `except ModuleNotFoundError` block when both `flash_attn` and `xformers` are absent.
**Root cause:** RTX 6000 compute capability 7.5 — `flash_attn` v2 requires sm_80+; xformers not installed. `LongNetConfig` hard-codes `flash_attention: True`.
**Status:** ✅ Patched

### 5. Slide embedding dim is 768, not 1536
The standard Prov-GigaPath slide encoder (`gigapath_slide_enc12l768d`) outputs `(768,)`, not `(1536,)` as stated in the architecture diagram and hyperparameters table. Update the Pathology Adapter input dim from 1536 to 768 when implementing `train_adapter.py`.

### 6. MI2 model substitution (rad-dino)
`microsoft/MedImageInsight` is not publicly available on HuggingFace (returns 404).
**Substitute used:** `microsoft/rad-dino` — DINOv2 ViT-B/14 trained on radiology images.
- Output dim: **768-d** CLS token per modality (not 1024-d as in MI2).
- Concatenated 4-modality embedding: **(3072,)** instead of (4096,).
- **Radiology Adapter input must be 3072**, not 4096.
- `scripts/mi2_encode.py` uses `rad-dino` via `transformers.AutoModel`. Smoke-tested ✅.

---

## Data Notes

- **Dataset:** TCGA-GBMLGG, 170 subjects
- **Grades:** 0 (40 subjects), 1 (53), 2 (77)
- **MRI modalities:** T1, T1-Post Contrast (T1-PC), T2, T2-FLAIR — each 512×512
- **WSI preprocessing:** ROI extraction from tumor region per Chen et al. IEEE TMI 2022; tile at 256×256 @ 0.5 mpp (1024×1024 native)
- **Clinical labels:** `survival_days` (continuous), `censored` (0/1), `grade` (0/1/2)

---

## Key Hyperparameters (from paper)

| Parameter | Value |
|-----------|-------|
| Tile size | 256×256 px |
| Tile resolution | 0.5 mpp |
| Radiology input | 512×512 per slice |
| PGP tile embedding dim | 1536 |
| PGP slide embedding dim | 1536 |
| MI2 embedding dim (per modality) | 1024 |
| Radiology concat dim | 4096 (4 × 1024) |
| Adapter hidden dim `d` | 512 (tunable) |
| Output embeddings dim | 512 |
| Fused embedding dim | 1024 (rad 512 + path 512) |
| Hazard output range | [-3, +3] |
| Loss | Cox partial log-likelihood |

---

## Useful Commands

```bash
# Check GPU allocation
squeue -u $USER

# Quick sanity check on a saved embedding
python -c "
import torch
e = torch.load('data/embeddings/PROV-000-000001_pgp_slide.pt')
print(e.shape, e.dtype)  # expect torch.Size([1536]) torch.float32
"

# Monitor GPU memory during encoding
watch -n1 nvidia-smi

# List tiles for a slide
ls data/tiles/PROV-000-000001.ndpi/ | wc -l
```

---

## References

1. Alberto Santamaria-Pang et al., "Cancer Survival with Radiology-Pathology Analysis and Healthcare AI Models in Azure AI Foundry," Microsoft Tech Community, Jan 2025.
2. Chen et al., "Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis," IEEE TMI 2022.
3. Can et al. (2022) — multimodal survival modeling reference.
4. Prov-GigaPath: https://aka.ms/provgigapathmodelcard
5. MedImageInsight (MI2): https://aka.ms/mi2modelcard
6. Sample code: https://aka.ms/healthcare-ai-examples-rad-path
