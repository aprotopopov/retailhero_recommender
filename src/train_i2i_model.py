import json
import os
import pickle
import sys

import implicit
import numpy as np
import pandas as pd
from scipy import sparse as sp
from tqdm import tqdm

import config as cfg
from utils import (
    ProductEncoder,
    get_shard_path,
    make_coo_row,
    normalized_average_precision,
)

if __name__ == "__main__":
    product_encoder = ProductEncoder(cfg.PRODUCT_CSV_PATH)

    rows = []
    for i in range(cfg.NUM_SHARDS - 1):
        for js in tqdm((json.loads(s) for s in open(get_shard_path(i)))):
            rows.append(
                make_coo_row(js["transaction_history"], product_encoder, normalize=True)
            )
    train_mat = sp.vstack(rows)

    model = implicit.nearest_neighbours.CosineRecommender(K=2)
    # model = implicit.nearest_neighbours.CosineRecommender(K=50)
    # model = implicit.nearest_neighbours.TFIDFRecommender(K=100)

    # ALS should be trained with normalize = False
    # model = implicit.als.AlternatingLeastSquares(factors=16, regularization=1e-5, iterations=12)
    model.fit(train_mat.T)

    out_dir = cfg.ASSETS_DIR
    os.makedirs(out_dir, exist_ok=True)
    print(f"Dump model to {out_dir}")
    pickle.dump(model, open(out_dir / "model.pkl", "wb"))

    print("Estimate quality...")
    scores = []
    for js in tqdm((json.loads(s) for s in open(get_shard_path(cfg.NUM_SHARDS - 1)))):
        row = make_coo_row(js["transaction_history"], product_encoder).tocsr()
        raw_recs = model.recommend(
            userid=0,
            user_items=row,
            N=30,
            filter_already_liked_items=False,
            recalculate_user=True,
        )

        recommended_items = product_encoder.toPid([idx for (idx, score) in raw_recs])
        gt_items = js["target"][0]["product_ids"]
        nap = normalized_average_precision(gt_items, recommended_items)
        scores.append(nap)
    print("nap: {}".format(np.mean(scores)))
