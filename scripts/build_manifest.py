"""
build_manifest.py — Scan embedding dirs and join with clinical labels to produce manifest.csv.

manifest.csv schema:
    subject_id, pgp_path, mi2_path, grade, survival_days, censored

clinical.csv must contain columns:
    subject_id, grade, survival_days, censored

Usage:
    python scripts/build_manifest.py \\
        --embedding-dir data/embeddings/ \\
        --clinical      data/raw/clinical.csv \\
        --out           data/manifest.csv
"""

import argparse
import os
import glob

import pandas as pd


def main(args):
    clinical = pd.read_csv(args.clinical)
    required_cols = {"subject_id", "grade", "survival_days", "censored"}
    missing = required_cols - set(clinical.columns)
    if missing:
        raise ValueError(f"clinical.csv missing columns: {missing}")

    emb_dir = args.embedding_dir

    rows = []
    for _, row in clinical.iterrows():
        sid = row["subject_id"]

        # Pathology embedding: <subject_id>_pgp_slide.pt
        pgp_candidates = glob.glob(os.path.join(emb_dir, f"{sid}*_pgp_slide.pt"))
        pgp_path = pgp_candidates[0] if pgp_candidates else None

        # Radiology embedding: <subject_id>_mi2.pt
        mi2_candidates = glob.glob(os.path.join(emb_dir, f"{sid}*_mi2.pt"))
        mi2_path = mi2_candidates[0] if mi2_candidates else None

        rows.append({
            "subject_id":    sid,
            "pgp_path":      pgp_path,
            "mi2_path":      mi2_path,
            "grade":         int(row["grade"]),
            "survival_days": float(row["survival_days"]),
            "censored":      int(row["censored"]),
        })

    df = pd.DataFrame(rows)

    n_total = len(df)
    n_pgp   = df["pgp_path"].notna().sum()
    n_mi2   = df["mi2_path"].notna().sum()
    n_both  = (df["pgp_path"].notna() & df["mi2_path"].notna()).sum()
    print(f"Subjects: {n_total}  |  pgp: {n_pgp}  |  mi2: {n_mi2}  |  both: {n_both}")

    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")

    if n_both == 0:
        print("WARNING: No subjects have both pgp and mi2 embeddings. "
              "Multimodal training is not yet possible.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-dir", required=True)
    ap.add_argument("--clinical",      required=True)
    ap.add_argument("--out",           required=True)
    main(ap.parse_args())
