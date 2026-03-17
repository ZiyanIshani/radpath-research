import os, re, glob
import torch
import timm
import numpy as np
import gigapath.torchscale.architecture.config as cfg
cfg.np = np
import gigapath.slide_encoder as slide_encoder
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

COORD_RE = re.compile(r"(\d+)x_(\d+)y\.(png|jpg|jpeg)$", re.IGNORECASE)

def parse_xy(path: str):
    m = COORD_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Bad tile filename (can't parse x/y): {path}")
    return int(m.group(1)), int(m.group(2))

def get_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

@torch.no_grad()
def main(tiles_dir: str, out_path: str, batch_size: int = 128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tile_paths = sorted(glob.glob(os.path.join(tiles_dir, "*.png")))
    if not tile_paths:
        tile_paths = sorted(glob.glob(os.path.join(tiles_dir, "*.jpg"))) + \
                     sorted(glob.glob(os.path.join(tiles_dir, "*.jpeg")))
    if not tile_paths:
        raise RuntimeError(f"No tiles found in {tiles_dir}")

    coords = torch.tensor([parse_xy(p) for p in tile_paths], dtype=torch.int32)

    tile_enc = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device)
    tile_enc.eval()

    slide_enc = slide_encoder.create_model(
        "hf_hub:prov-gigapath/prov-gigapath",
        "gigapath_slide_enc12l768d",
        1536,
        global_pool=True
    ).to(device)
    slide_enc.eval()

    tfm = get_transform()

    embs = []
    for i in tqdm(range(0, len(tile_paths), batch_size), desc="Encoding tiles"):
        imgs = []
        for p in tile_paths[i:i+batch_size]:
            with open(p, "rb") as f:
                imgs.append(tfm(Image.open(f).convert("RGB")))
        x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
        y = tile_enc(x)  # (B,1536)
        embs.append(y.detach().cpu())

    tile_embed = torch.cat(embs, dim=0)                 # (N,1536)
    slide_out = slide_enc(tile_embed.unsqueeze(0).to(device), coords.unsqueeze(0).to(device))
    slide_vec = slide_out[0].squeeze().detach().cpu()  # (1536,) typically

    torch.save({
        "slide_embedding": slide_vec,
        "tile_embed": tile_embed,
        "coords": coords,
        "tiles_dir": tiles_dir,
    }, out_path)

    print("tile_embed:", tuple(tile_embed.shape))
    print("coords:", tuple(coords.shape))
    print("slide_embedding:", tuple(slide_vec.shape))
    print("saved:", out_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    main(args.tiles_dir, args.out, args.batch_size)