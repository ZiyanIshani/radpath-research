# Rad-Path Survival Prediction — Implementation Status

Reimplementation of the Microsoft Azure AI Foundry rad-path survival pipeline
([blog post](https://techcommunity.microsoft.com/blog/healthcareandlifesciencesblog/cancer-survival-with-radiology-pathology-analysis-and-healthcare-ai-models-in-az/4366241))
using locally-run foundation models on the Phoenix HPC cluster (Georgia Tech PACE).

**Goal:** Predict cancer grade and patient survival from fused radiology (MRI) +
histopathology (H&E WSI) embeddings using the TCGA-GBMLGG glioma dataset.

---

## Dataset

**TCGA-GBMLGG** — 170 subjects, brain glioma grades 0–2.

| Grade | Subjects |
|-------|----------|
| 0 (low-grade astrocytoma) | 40 |
| 1 (oligodendroglioma)     | 53 |
| 2 (glioblastoma)          | 77 |

**Clinical labels per subject:** `survival_days` (continuous), `censored` (0/1 event indicator), `grade` (0/1/2).

**MRI modalities per subject:** T1, T1-Post Contrast (T1-PC), T2, T2-FLAIR — each volume stored as NIfTI (`.nii.gz`).

**WSI:** One `.ndpi` whole-slide image per subject.

---

## Architecture

```
Radiology MRI (T1, T1-PC, T2, T2-FLAIR)
        │
        ▼
  rad-dino (DINOv2 ViT-B/14, radiology-pretrained)        [microsoft/rad-dino]
  Applied independently to center axial slice of each modality
        │
        ▼
  CLS token per modality: (768,)  ×  4 modalities
        │
        ▼  concatenate
  Radiology embedding: (3072,)
        │
        ▼
  Radiology Adapter: Linear(3072, d) → ReLU → Linear(d, 512)
        │
        ▼
  Radiology output: (512,)  ──────────────────────────────┐
                                                           │
                                                    [concat → (1024,)]
                                                           │
                                                 Multi-Modal Adapter:
                                                 Linear(1024, d) → ReLU → Linear(d, 1)
                                                           │
Histopathology WSI (.ndpi)                                 ▼
        │                                          Hazard scalar ∈ [-3, +3]
        ▼  tile 256×256 @ 0.5 mpp
  N tiles (PNG), ~1000–2000 per slide
        │
        ▼
  GigaPath tile encoder (ViT-g/14)                 [prov-gigapath/prov-gigapath]
  Applied to all N tiles in batches
        │
        ▼
  Tile embeddings: (N, 1536)
  Coordinates: (N, 2)  [x, y parsed from filenames]
        │
        ▼
  GigaPath slide encoder (gigapath_slide_enc12l768d, 12-layer LongNet)
        │
        ▼
  Slide embedding: (768,)
        │
        ▼
  Pathology Adapter: Linear(768, d) → ReLU → Linear(d, 512)
        │
        ▼
  Pathology output: (512,)  ─────────────────────────────┘
```

**Loss function:** Cox proportional hazards loss (Breslow approximation for ties).
**Metric:** Concordance index (C-index), Kaplan-Meier curves per grade.
**Hazard output:** Scalar in `[-3, +3]` — negative = low risk, positive = high risk.

---

## Corrected Dimensionalities (vs. Original Paper)

The original paper/blog describes MI2 and GigaPath dimensions that differ from what
our substitute models actually produce. The table below shows **what the code actually uses**:

| Stage | Paper claim | Actual (this impl.) | Note |
|-------|------------|---------------------|------|
| Tile encoder output | (N, 1536) | **(N, 1536)** | Matches — GigaPath ViT-g/14 |
| Slide encoder output | (1536,) | **(768,)** | `gigapath_slide_enc12l768d` outputs 768-d, not 1536-d |
| MI2 per-modality output | (1024,) | **(768,)** | rad-dino substitute; MI2 not publicly available |
| Radiology concat (4 modalities) | (4096,) | **(3072,)** | 4 × 768 = 3072 |
| Radiology Adapter input | Linear(4096, ...) | **Linear(3072, ...)** | corrected |
| Pathology Adapter input | Linear(1536, ...) | **Linear(768, ...)** | corrected |
| Adapter hidden dim `d` | 512 | **512** (tunable) | default |
| Fused embedding dim | 1024 | **1024** | 512 + 512 — unchanged |
| Hazard output | scalar | **scalar ∈ [-3, +3]** | clamped |

---

## What Has Been Verified

### Step 1 — WSI Tiling (`gigapath_wsi` conda env)

**Script:** `scripts/tile_wsi.py`

**Verified on:** 1 slide
**Slide:** `PROV-000-000001.ndpi`
**Output location:** `data/tiles/PROV-000-000001.ndpi/`
**Number of tiles produced:** **1,070 tiles**
**Tile format:** 256×256 PNG, filenames encode spatial coordinates as `<x>x_<y>y.png`
(e.g., `03072x_12416y.png`).

The tiler uses OpenSlide to find the resolution level closest to 0.5 mpp and extracts
non-background 256×256 patches. Typical yield is 1,000–2,000 tiles per glioma WSI.

### Step 2 — Tile + Slide Encoding (`.venv`, GPU)

**Script:** `scripts/pgp_encode_from_tiles.py`

**Verified on:** 1 slide (`PROV-000-000001`)
**Output file:** `data/embeddings/PROV-000-000001_pgp_slide.pt`

**What happens internally:**

1. All 1,070 tile PNGs are loaded, resized to 256→center-cropped to 224, normalized
   (ImageNet stats), and batched (default batch size 128).
2. **Tile encoder** (`hf_hub:prov-gigapath/prov-gigapath`, ViT-g/14) produces embeddings
   of shape `(1070, 1536)` on the GPU.
3. `(x, y)` coordinates are parsed from filenames into a `(1070, 2)` int32 tensor.
4. Both tensors are unsqueezed to add a batch dimension: `(1, 1070, 1536)` and `(1, 1070, 2)`.
5. **Slide encoder** (`gigapath_slide_enc12l768d`, 12-layer LongNet transformer with
   global pooling) aggregates the tile embeddings using spatial positions and outputs
   a list; element `[0]` is squeezed to shape `(768,)`.
6. The `.pt` file saves a dict with keys: `slide_embedding` `(768,)`, `tile_embed`
   `(1070, 1536)`, `coords` `(1070, 2)`, `tiles_dir`.

**Model weights location:** `weights/pgp/` (files: `pytorch_model.bin`, `slide_encoder.pth`, `config.json`).

### Step 3 — Radiology Encoding (`.venv`, GPU)

**Script:** `scripts/mi2_encode.py`

**Verified on:** Synthetic NIfTI volumes (smoke test on GPU — no real MRI data ingested yet)

**Substitute model:** `microsoft/rad-dino` (DINOv2 ViT-B/14 trained on radiology images).
The original paper uses `microsoft/MedImageInsight` (MI2), which is **not publicly
available** on HuggingFace (returns 404).

**What happens internally:**

1. For each of the 4 MRI modalities (T1, T1-PC, T2, T2-FLAIR):
   - Load the NIfTI volume with `nibabel`.
   - Extract the center axial slice (along the longest spatial dimension).
   - Clip intensities to [1st, 99th] percentile, normalize to [0, 255].
   - Convert grayscale to 3-channel RGB PIL image.
   - Run through `rad-dino` processor + model.
   - Extract CLS token: `last_hidden_state[:, 0, :]` → shape `(1, 768)` → squeeze to `(768,)`.
2. Concatenate in canonical order [T1, T1-PC, T2, FLAIR] → `(3072,)`.
3. Save dict to `.pt` with keys: `mi2_embedding` `(3072,)`, `modality_embs` (dict of 4× `(768,)`),
   `subject_id`, `model_id`, `modality_order`.

**Output file naming:** `data/embeddings/<subject_id>_mi2.pt`

---

## What Still Needs to Be Done

### 1. Obtain real TCGA-GBMLGG MRI data

- Download NIfTI MRI volumes from TCIA (The Cancer Imaging Archive) for all 170 subjects.
- Place under `data/raw/mri/<subject_id>/{T1,T1PC,T2,FLAIR}.nii.gz`.
- Run `scripts/mi2_encode.py` per subject to produce `data/embeddings/<subject_id>_mi2.pt`.
- **Note:** The radiology encoder script and output format are ready — it just needs real inputs.

### 2. Run tiling + PGP encoding for all 170 WSIs

- Only 1 slide (`PROV-000-000001`) has been tiled and encoded so far.
- Need to run `scripts/tile_wsi.py` (in `gigapath_wsi` env) then
  `scripts/pgp_encode_from_tiles.py` (in `.venv` on GPU) for each `.ndpi`.
- May benefit from a batch wrapper script.

### 3. Build manifest (`scripts/build_manifest.py`)

- Script exists and is ready.
- Blocked on having real clinical labels (`data/raw/clinical.csv` with columns
  `subject_id, grade, survival_days, censored`) and actual embeddings for all subjects.
- Run after steps 1 and 2 are complete:
  ```bash
  python scripts/build_manifest.py \
      --embedding-dir data/embeddings/ \
      --clinical      data/raw/clinical.csv \
      --out           data/manifest.csv
  ```

### 4. Train adapters (`scripts/train_adapter.py`)

- Script is **fully written** with correct dimensions (768 path-in, 3072 rad-in).
- Implements Cox loss (Breslow), C-index, 80/20 train/val split, cosine LR decay.
- Blocked on `manifest.csv` existing with valid embedding paths.
- Run:
  ```bash
  python scripts/train_adapter.py \
      --manifest data/manifest.csv \
      --mode multimodal \
      --d 512 \
      --epochs 100 \
      --lr 1e-4 \
      --out models/checkpoints/multimodal_v1.pt
  ```
- `--mode` options: `radiology` | `pathology` | `multimodal`

### 5. Evaluate (`scripts/evaluate.py`)

- Script does **not yet exist**.
- Should load a saved checkpoint, run inference on held-out subjects, compute C-index,
  and plot Kaplan-Meier curves stratified by grade.

---

## Current Status Summary

| Component | Status | Details |
|-----------|--------|---------|
| GPU environment (`.venv`) | Done | PyTorch 2.8+cu128, Quadro RTX 6000, CUDA verified |
| WSI tiling | Done (1/170) | `PROV-000-000001.ndpi` → 1,070 tiles in `data/tiles/` |
| PGP tile encoder | Done (1/170) | `(1070, 1536)` tile embeddings |
| PGP slide encoder | Done (1/170) | `PROV-000-000001_pgp_slide.pt`, `slide_embedding` shape `(768,)` |
| Radiology encoder (rad-dino) | Smoke-tested | Script ready; verified with synthetic NIfTI on GPU |
| Real MRI data | Not acquired | Must download from TCIA for all 170 subjects |
| Batch encoding pipeline | Not started | Need loop over all 170 subjects for both modalities |
| Manifest builder | Ready (not run) | Waiting on real embeddings + clinical CSV |
| Adapter training | Ready (not run) | `train_adapter.py` written; waiting on manifest |
| Evaluation | Not started | `evaluate.py` does not exist yet |

---

## Known Patches Applied to External Code

### 1. `external/prov-gigapath/gigapath/torchscale/architecture/config.py`
**Problem:** `NameError: name 'np' is not defined` — the config uses `eval()` on strings
containing `np.*` expressions but numpy was not imported.
**Fix:** Added `import numpy as np` at the top of the file.

### 2. `scripts/pgp_encode_from_tiles.py` (line 67) — batch dimension
**Problem:** Slide encoder `forward()` expects batch-first tensors `(1, N, D)` and returns
a list, but the script passed `(N, D)` and treated the output as a tensor.
**Fix:** Added `.unsqueeze(0)` to both `tile_embed` and `coords` before calling the slide
encoder; added `[0]` indexing to unpack the returned list.

### 3. `external/prov-gigapath/gigapath/torchscale/component/flash_attention.py`
**Problem:** `LongNetConfig` hard-codes `flash_attention: True`, but the RTX 6000 (compute
capability 7.5) cannot run `flash_attn` v2 (requires sm_80+), and xformers is not installed.
**Fix:** Added a standard PyTorch `torch.bmm`-based attention fallback in the
`except ModuleNotFoundError` block so the slide encoder runs without flash attention.

---

## Compute Environment

**Cluster:** Phoenix (Georgia Tech PACE)
**GPU:** Quadro RTX 6000, 24 GB VRAM, compute capability 7.5

```bash
# Request GPU node
srun --partition=gpu-rtx6000 --gres=gpu:1 --pty bash

# Activate GPU/torch env (for encoding and training)
source ~/scratch/radpath/.venv/bin/activate

# Activate tiling env (for WSI → tiles only)
conda activate gigapath_wsi
```

**Two environments — do not mix:**

| Env | Purpose | Key packages |
|-----|---------|-------------|
| `conda gigapath_wsi` | WSI tiling (OpenSlide, ndpi support) | openslide, pillow |
| `~/scratch/radpath/.venv` | GPU embedding + training | PyTorch 2.8, timm, transformers, nibabel |

Any script calling `torch`, `timm`, or the GigaPath / rad-dino models must use `.venv` on a GPU node.

---

## Repository Layout

```
radpath/
├── data/
│   ├── raw/                          # (empty — real data not yet downloaded)
│   │   ├── wsi/                      # .ndpi whole-slide images
│   │   ├── mri/                      # per-subject NIfTI volumes
│   │   └── clinical.csv              # subject_id, grade, survival_days, censored
│   ├── tiles/
│   │   └── PROV-000-000001.ndpi/     # 1,070 × 256×256 PNG tiles (verified)
│   └── embeddings/
│       └── PROV-000-000001_pgp_slide.pt   # slide_embedding (768,) + tile_embed (1070,1536)
├── external/
│   └── prov-gigapath/                # cloned GigaPath repo (3 patches applied)
├── scripts/
│   ├── tile_wsi.py                   # WSI → tiles (gigapath_wsi env)
│   ├── pgp_encode_from_tiles.py      # tiles → slide embedding (.venv + GPU)
│   ├── mi2_encode.py                 # MRI → radiology embedding (.venv + GPU)
│   ├── build_manifest.py             # scan embeddings + clinical CSV → manifest.csv
│   ├── train_adapter.py              # Cox survival training (.venv + GPU)
│   └── evaluate.py                   # NOT YET WRITTEN
├── weights/
│   └── pgp/
│       ├── pytorch_model.bin         # GigaPath tile encoder weights
│       ├── slide_encoder.pth         # GigaPath slide encoder weights
│       └── config.json
└── CLAUDE.md                         # developer notes and patch log
```

---

## References

1. Santamaria-Pang et al., "Cancer Survival with Radiology-Pathology Analysis and Healthcare AI Models in Azure AI Foundry," Microsoft Tech Community, Jan 2025.
2. Chen et al., "Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis," IEEE TMI 2022.
3. Prov-GigaPath: https://aka.ms/provgigapathmodelcard
4. rad-dino (MI2 substitute): `microsoft/rad-dino` on HuggingFace
