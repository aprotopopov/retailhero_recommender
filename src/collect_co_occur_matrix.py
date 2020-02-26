import json
from pathlib import Path

import numpy as np
from scipy import sparse as sp
from tqdm import tqdm

import config as cfg
from utils import ProductEncoder, get_shard_path


def collect_cooccur_matrix(shard_indices, product_encoder):
    num_products = product_encoder.num_products
    co_occurrence = np.zeros((num_products, num_products))
    occurrence = np.zeros(num_products)
    for shard_idx in tqdm(shard_indices):
        for js in tqdm((json.loads(s) for s in open(get_shard_path(shard_idx)))):
            tids = js.get("transaction_history", [])
            for tid in tids:
                product_ind = [
                    product_encoder.toIdx(item["product_id"])
                    for item in tid.get("products", [])
                ]
                for pid_num, pid in enumerate(product_ind):
                    occurrence[pid] += 1
                    for co_pid in product_ind[pid_num + 1 :]:
                        co_occurrence[co_pid][pid] += 1
                        co_occurrence[pid][co_pid] += 1
    return co_occurrence, occurrence


if __name__ == "__main__":
    product_encoder = ProductEncoder(cfg.PRODUCT_CSV_PATH)
    co_occurrence, occurrence = collect_cooccur_matrix(
        range(cfg.NUM_SHARDS), product_encoder
    )

    # cut the low count records to reduce size and improve speed
    min_count = 5
    co_occurrence_sp = sp.csc_matrix(
        np.where(co_occurrence >= min_count, co_occurrence, 0), dtype=np.int32
    )

    # not compressed for fast loading
    sp.save_npz(
        cfg.ASSETS_DIR / f"item_co_occurrence_min_cnt_{min_count}.npz",
        co_occurrence_sp,
        compressed=False,
    )
    np.save(cfg.ASSETS_DIR / "item_occurrence.npy", occurrence)
