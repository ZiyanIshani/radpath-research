import os, glob, re
import torch
import timm
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from gigapath.preprocessing.data.slide_utils import find_level_for_target_mpp
from gigapath.pipeline import tile_one_slide
import gigapath.slide_encoder as slide_encoder


# --- helpers ---
def parse_xy_from_filename(path: str):
    """
    Supports common Prov-GigaPath tile naming patterns, e.g.:
      '256x_512y.png'  OR  'x256_y512.png'  OR  '256x_512y.jpeg'
    We only need (x, y) in slide coordinate space used during tiling.
    """
    name = os.path.basename(path)

    # pattern like 256x_512y.png
    m = re.search(r"(\d+)x[_-](\d+)y", name)
    if m:
        return int(m.group(1)), int(m.group(2))

    # pattern like x256_y512.png
    m = re.search(r"x(\d+)[_-]y(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))

    raise ValueError(f"Could not parse coords from tile filename: {name}")


class TileDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        x, y = parse_xy_from_filename(p)
        with open(p, "rb") as f:
            img = Image.open(f).convert("RGB")
        t = self.transform(img)  # (3,224,224)
        return t, torch.tensor([x, y], dtype=torch.int32)


def get_transform():
    # Matches README for tile encoder inference. :contentReference[oaicite:4]{index=4}
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def main(slide_path: str, out_path: str, work_dir: str, target_mpp: float = 0.5, batch_size: int = 128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Slide: {slide_path}")
    print(f"Target mpp: {target_mpp} (model trained w/ 0.5 mpp guidance in README)")

    # 1) choose level for ~0.5 mpp
    level = find_level_for_target_mpp(slide_path, target_mpp)
    if level is None:
        raise RuntimeError("No suitable level found for target mpp. Try a different target_mpp or inspect slide pyramid.")
    print("Using level:", level)

    # 2) tile the slide
    # tile_one_slide saves tiles into work_dir; README calls out tiling + coords pipeline. :contentReference[oaicite:5]{index=5}
    print("Tiling...")
    tile_one_slide(slide_path, save_dir=work_dir, level=level)

    # 3) gather tiles
    tile_paths = sorted(glob.glob(os.path.join(work_dir, "**", "*.png"), recursive=True))
    if len(tile_paths) == 0:
        # sometimes tiles may be jpg
        tile_paths = sorted(glob.glob(os.path.join(work_dir, "**", "*.jpg"), recursive=True)) + \
                     sorted(glob.glob(os.path.join(work_dir, "**", "*.jpeg"), recursive=True))
    if len(tile_paths) == 0:
        raise RuntimeError(f"No tiles found under {work_dir}. Check tiling output.")

    print(f"Found {len(tile_paths)} tiles")

    # 4) load tile encoder (timm HF hub)
    # README uses timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True). :contentReference[oaicite:6]{index=6}
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device)
    tile_encoder.eval()

    tfm = get_transform()
    ds = TileDataset(tile_paths, tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 5) encode tiles
    all_embs = []
    all_coords = []
    print("Encoding tiles...")
    for x, coords in tqdm(dl):
        x = x.to(device, non_blocking=True)
        y = tile_encoder(x)             # (B,1536) per your successful test
        all_embs.append(y.detach().cpu())
        all_coords.append(coords.cpu())

    tile_embed = torch.cat(all_embs, dim=0)      # (N,1536)
    coordinates = torch.cat(all_coords, dim=0)   # (N,2)

    print("tile_embed:", tuple(tile_embed.shape), "coords:", tuple(coordinates.shape))

    # 6) load slide encoder + run
    # README shows create_model(..., "gigapath_slide_enc12l768d", 1536). :contentReference[oaicite:7]{index=7}
    slide_enc = slide_encoder.create_model(
        "hf_hub:prov-gigapath/prov-gigapath",
        "gigapath_slide_enc12l768d",
        1536,
        global_pool=True
    ).to(device)
    slide_enc.eval()

    print("Running slide encoder...")
    out = slide_enc(tile_embed.to(device), coordinates.to(device)).detach().cpu()

    # out shape depends on global_pool; expected a single slide vector
    print("slide_out:", tuple(out.shape))

    # 7) save
    torch.save({
        "slide_embedding": out.squeeze(),   # (1536,) typically
        "tile_embed": tile_embed,           # keep for debugging/ablation
        "coords": coordinates,
        "slide_path": slide_path,
        "level": int(level),
        "target_mpp": float(target_mpp),
    }, out_path)

    print("Saved:", out_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--target_mpp", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    main(args.slide, args.out, args.work_dir, args.target_mpp, args.batch_size)