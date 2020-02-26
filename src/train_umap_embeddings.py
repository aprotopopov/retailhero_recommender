import json
import logging

import numpy as np
import pandas as pd
import umap
from scipy import sparse as sp
from tqdm import tqdm

import config as cfg
from utils import ProductEncoder, get_shard_path, make_coo_row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def get_train_data(max_rows=None):
    product_encoder = ProductEncoder(cfg.PRODUCT_CSV_PATH)

    rows = []
    num_rows = 0
    for shard_idx in tqdm(range(cfg.NUM_SHARDS)):
        for js in tqdm(json.loads(s) for s in open(get_shard_path(shard_idx))):
            rows.append(
                make_coo_row(js["transaction_history"], product_encoder, normalize=True)
            )
            num_rows += 1

            if max_rows and num_rows == max_rows:
                return sp.vstack(rows)

    trans_mat = sp.vstack(rows)
    return trans_mat


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    UMAP_MAX_ROWS = 1000
    trans_mat = get_train_data(UMAP_MAX_ROWS)

    umap_params = dict(
        random_state=14, metric="cosine", n_neighbors=10, low_memory=True
    )

    logger.info("Training UMAP embeddings.")
    umap_items = umap.UMAP(**umap_params)
    item_embeddings = umap_items.fit_transform(trans_mat.T.tocsr())

    filename = cfg.ASSETS_DIR / "umap_item_emb.npy"
    logger.info(f"Saving UMAP embeddings to {filename}")
    np.save(filename, item_embeddings)
