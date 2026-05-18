#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp

import anndata as ad
import muon as mu
from muon import MuData

MODALITY_NAMES: dict[str, str] = {
    "Gene Expression": "rna",
    "CRISPR Guide Capture": "crispr",
}

ANNOTATION_URL = (
    "https://ars.els-cdn.com/content/image/1-s2.0-S0092867422005979-mmc1.xlsx"
)

DATASETS = {
    "K562_essential": {
        "figshare_id": "20127869",
        "filename_pattern": "K562_essential*",
        "prefix_regex": r"(KD6_\d+_essential_)matrix\.mtx\.gz",
        "annotation_sheet": "TabA_K562_day8_library",
    },
    "K562_gwps": {
        "figshare_id": "20127869",
        "filename_pattern": "K562_gwps*",
        "prefix_regex": r"(KD8_\d+_gwps_)matrix\.mtx\.gz",
        "annotation_sheet": "TabB_K562_day6_library",
    },
    "rpe1": {
        "figshare_id": "20127869",
        "filename_pattern": "rpe1*",
        "prefix_regex": r"(RD7_\d+_)matrix\.mtx\.gz",
        "annotation_sheet": "TabC_RPE1_day7_library",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Perturb-seq MEX data and convert to MuData"
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=list(DATASETS.keys()),
        help="dataset to download",
    )
    parser.add_argument("--output_dir", "-o", required=True, help="output directory")
    parser.add_argument("--name", "-n", required=True, help="output file prefix")
    return parser.parse_args()


def fetch(
    figshare_id: str, filename_pattern: str, out_dir: str | os.PathLike[str]
) -> str:
    """Download from Figshare via hapiq. Returns path to the downloaded tar.gz.

    hapiq always creates {out_dir}/{figshare_id}/ and places files there.
    """
    print(f"Downloading figshare {figshare_id} ...", file=sys.stderr)
    subprocess.run(
        [
            "hapiq",
            "download",
            "figshare",
            figshare_id,
            "-q",
            "--timeout",
            str(60 * 60),  # 1 hour
            "--out",
            out_dir,
            "--filename-pattern",
            filename_pattern,
        ],
        check=True,
    )
    download_dir = os.path.join(out_dir, figshare_id)
    tars = [f for f in os.listdir(download_dir) if f.endswith(".tar.gz")]
    if not tars:
        raise FileNotFoundError(f"No .tar.gz found in {download_dir}")
    return os.path.join(download_dir, tars[0])


def find_mex_dir(extract_dir: str | os.PathLike[str]) -> str:
    """Walk extract_dir and return the first directory containing MEX matrix files."""
    for root, _, files in os.walk(extract_dir):
        if any(f.endswith("_matrix.mtx.gz") for f in files):
            return root
    raise FileNotFoundError("No MEX files found after extraction")


def read_batches(
    mex_dir: str | os.PathLike[str], prefix_regex: str, n_workers: int = 8
) -> MuData:
    """Read all per-batch MEX directories, annotate obs with batch ID, and concatenate into a single MuData."""
    prefixes = sorted(
        m.group(1) for f in os.listdir(mex_dir) if (m := re.match(prefix_regex, f))
    )
    if not prefixes:
        raise ValueError(f"No batch prefixes matched in {mex_dir}")

    print(f"  found {len(prefixes)} batches", file=sys.stderr)

    def _read_one(prefix: str) -> MuData:
        mdata = mu.read_10x_mtx(mex_dir, prefix=prefix)
        batch_id = re.search(r"_(\d+)_", prefix).group(1)
        for mod in mdata.mod:
            mdata[mod].obs["batch"] = batch_id
        return mdata

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        mdatas = list(executor.map(_read_one, prefixes))

    mdata = mu.MuData(
        {
            MODALITY_NAMES.get(mod, mod): ad.concat(
                [m[mod] for m in mdatas], index_unique="-"
            )
            for mod in mdatas[0].mod.keys()
        }
    )
    for mod in mdata.mod:
        print(f"  {mod}: {mdata[mod].n_obs} x {mdata[mod].n_vars}", file=sys.stderr)
    return mdata


def fetch_annotation(out_dir: str | os.PathLike[str], sheet: str) -> pd.DataFrame:
    """Download the Replogle library annotation and return a guide_id → (pair_id, gene) table."""
    xlsx_path = os.path.join(out_dir, "replogle_annotation.xlsx")
    if not os.path.exists(xlsx_path):
        print("Downloading library annotation ...", file=sys.stderr)
        subprocess.run(["curl", "-fsSL", "-o", xlsx_path, ANNOTATION_URL], check=True)
    anno = pd.read_excel(xlsx_path, sheet_name=sheet)
    return pd.concat(
        [
            anno[["unique sgRNA pair ID", "gene", "sgID_A"]].rename(
                columns={"sgID_A": "guide_id"}
            ),
            anno[["unique sgRNA pair ID", "gene", "sgID_B"]].rename(
                columns={"sgID_B": "guide_id"}
            ),
        ]
    ).set_index("guide_id")


def aggregate_guide_pairs(mdata: MuData, anno: pd.DataFrame) -> None:
    """Sum counts per guide pair using the library annotation."""
    if "crispr" not in mdata.mod:
        return
    crispr = mdata["crispr"]

    # Strip _posA / _posB suffix to match annotation guide IDs
    guide_ids = crispr.var_names.str[:-5]
    joined = pd.DataFrame({"guide_id": guide_ids}).join(anno, on="guide_id", how="left")
    pair_ids = joined["unique sgRNA pair ID"].fillna(pd.Series(guide_ids, dtype=str))

    unique_pairs, inverse = np.unique(pair_ids.values, return_inverse=True)
    n_guides, n_pairs = len(pair_ids), len(unique_pairs)
    indicator = sp.csr_matrix(
        (np.ones(n_guides), (np.arange(n_guides), inverse)),
        shape=(n_guides, n_pairs),
    )

    pair_gene = (
        joined[["unique sgRNA pair ID", "gene"]]
        .drop_duplicates("unique sgRNA pair ID")
        .set_index("unique sgRNA pair ID")["gene"]
    )
    var = pd.DataFrame(
        {"target_gene": pair_gene.reindex(unique_pairs).values},
        index=pd.Index(unique_pairs, name="pair_id"),
    )

    mdata.mod["crispr"] = ad.AnnData(
        X=crispr.X @ indicator,
        obs=crispr.obs.copy(),
        var=var,
    )
    mdata.update()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = DATASETS[args.dataset_name]

    tmp = os.path.join(args.output_dir, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    try:
        tar_path = fetch(cfg["figshare_id"], cfg["filename_pattern"], tmp)

        print(f"Extracting {os.path.basename(tar_path)} ...", file=sys.stderr)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp)

        print("Reading MEX batches ...", file=sys.stderr)
        mdata = read_batches(find_mex_dir(tmp), cfg["prefix_regex"])
    finally:
        shutil.rmtree(tmp)

    anno = fetch_annotation(args.output_dir, cfg["annotation_sheet"])
    aggregate_guide_pairs(mdata, anno)

    out_path = os.path.join(args.output_dir, f"{args.name}.h5mu")
    mdata.write(out_path)


if __name__ == "__main__":
    main()
