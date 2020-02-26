import json
from collections import defaultdict

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


def update_item_cost(transaction_history, product_encoder, storage):
    for txn in transaction_history:
        for item in txn["products"]:
            key = product_encoder.toIdx(item["product_id"])
            item_cost = item["s"] / max(item["quantity"], 1)
            if storage[key] == 0:
                storage[key] = item_cost
            else:
                storage[key] = (storage[key] + item_cost) / 2.0


if __name__ == "__main__":
    product_encoder = ProductEncoder(cfg.PRODUCT_CSV_PATH)
    num_products = product_encoder.num_products

    items_cost = defaultdict(int)
    rows = []
    num_transactions = 0
    for i in tqdm(range(cfg.NUM_SHARDS)):
        for js in tqdm((json.loads(s) for s in open(get_shard_path(i)))):
            update_item_cost(js["transaction_history"], product_encoder, items_cost)
            rows.append(
                make_coo_row(
                    js["transaction_history"], product_encoder, normalize=False
                )
            )
            num_transactions += len(js["transaction_history"])
    trans_mat = sp.vstack(rows)

    items_cnt = trans_mat.sum(axis=0).A[0]
    df_top_items = (
        pd.Series(items_cnt, name="items_cnt").sort_values(ascending=False).to_frame()
    )
    df_items_cost = pd.Series(items_cost, name="cost").to_frame()
    df_misc_features = df_top_items.join(df_items_cost)
    df_misc_features["popularity_position"] = range(num_products)

    df_misc_features.to_csv(cfg.ASSETS_DIR / "products_misc.csv")

    # iDF, products in user purchases
    bought_products = trans_mat.sum(axis=0).A[0]
    user_prod_log_idf = np.log(trans_mat.shape[0] / (bought_products + 1))
    np.save(cfg.ASSETS_DIR / "user_prod_log_idf.npy", user_prod_log_idf)

    # iDF, products in transactions
    tran_prod_log_idf = np.log(num_transactions / (bought_products + 1))
    np.save(cfg.ASSETS_DIR / "tran_prod_log_idf.npy", tran_prod_log_idf)
