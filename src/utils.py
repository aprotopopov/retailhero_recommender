import json
import hashlib
from typing import List, Set
from collections import defaultdict

import feather
import numpy as np
import pandas as pd
from scipy import sparse as sp

import config as cfg


class ProductEncoder:
    def __init__(self, product_csv_path):
        self.product_idx = {}
        self.product_pid = {}
        for idx, pid in enumerate(pd.read_csv(product_csv_path).product_id.values):
            self.product_idx[pid] = idx
            self.product_pid[idx] = pid

    def toIdx(self, x):
        if type(x) == str:
            pid = x
            return self.product_idx[pid]
        return [self.product_idx[pid] for pid in x]

    def toPid(self, x):
        if type(x) == int:
            idx = x
            return self.product_pid[idx]
        return [self.product_pid[idx] for idx in x]

    @property
    def num_products(self):
        return len(self.product_idx)


class ProductFeatEncoder:
    def __init__(self, product_csv_path, init=True, exclude=None):
        # self.products = pd.read_csv(product_csv_path).fillna(0)
        self.products = pd.read_csv(product_csv_path)
        self.products_map = defaultdict(dict)
        self.features = defaultdict(dict)
        if exclude:
            self.products = self.products.drop(exclude, axis=1)

        if init:
            self.create_encoder()
            self.create_features()

    def create_encoder(self):
        cols = self.products.dtypes[self.products.dtypes == "object"].index.tolist()
        global_idx = 0
        for col in cols:
            self.products_map[col] = defaultdict(dict)
            for idx, pid in enumerate(self.products[col].unique()):
                self.products_map[col]["pid"][pid] = idx
                self.products_map[col]["idx"][idx] = pid
                self.products_map[col]["gidx"][idx] = pid

    def create_features(self):
        for tup in self.products.set_index("product_id").itertuples():
            for name, val in tup._asdict().items():
                if name == "Index":
                    continue

                val = (
                    self.products_map[name]["pid"][val]
                    if name in self.products_map
                    else val
                )
                self.features[tup.Index][name] = val

    def product_features(self, ids):
        if type(ids) == str:
            return self.features[ids]
        return [self.features[product_id] for product_id in ids]

    def product_features_idx(self, ind):
        if type(ind) == str:
            return self.features[ind]
        return [
            self.features[self.products_map["product_id"]["idx"][idx]] for idx in ind
        ]


class TrainingSample:
    def __init__(
        self, row: sp.coo_matrix, target_items: Set[int], client_id: str = None
    ):
        self.row = row
        self.target_items = target_items
        self.client_id = client_id


def make_coo_row(
    transaction_history,
    product_encoder: ProductEncoder,
    last_transaction=False,
    normalize=True,
    entity=False,
):
    idx = []
    values = []

    items = defaultdict(int)
    if last_transaction:
        transaction_history = transaction_history[-1:]

    for trans in transaction_history:
        for i in trans["products"]:
            pidx = product_encoder.toIdx(i["product_id"])
            items[pidx] += 1.0
    n_items = sum(items.values())

    for pidx, val in items.items():
        idx.append(pidx)
        if normalize:
            val = val / n_items
        if entity:
            val = 1
        values.append(val)

    return sp.coo_matrix(
        (np.array(values).astype(np.float32), ([0] * len(idx), idx)),
        shape=(1, product_encoder.num_products),
    )


def cache_to_feather(cache_fm, num_products=43038):
    dfs = []
    for key, item in cache_fm.items():
        df_model = pd.Series(item).apply(pd.Series)
        dfs.append(df_model)

    df_scores = dfs[0]
    for df in dfs[1:]:
        df_scores = df_scores.join(df)

    df_scores = df_scores.sort_index().reindex(range(num_products)).fillna(0)
    feather.write_dataframe(df_scores, cfg.ASSETS_DIR / "implicit_scores.feather")


def create_products_in_transaction(
    transaction_history, product_encoder: ProductEncoder, outfile
):
    """Collect item2vec file."""
    for trans in transaction_history:
        products_str = " ".join(
            str(product_encoder.toIdx(i["product_id"])) for i in trans["products"]
        )
        outfile.write(products_str + "\n")
    outfile.flush()


def update_item_cost(transaction_history, product_encoder, storage):
    for txn in transaction_history:
        for item in txn["products"]:
            key = product_encoder.toIdx(item["product_id"])
            item_cost = item["s"] / max(item["quantity"], 1)

            if storage[key] == 0:
                storage[key] = item_cost
            else:
                storage[key] = (storage[key] + item_cost) / 2.0


def average_precision(actual, recommended, k=30):
    ap_sum = 0
    hits = 0
    for i in range(k):
        product_id = recommended[i] if i < len(recommended) else None
        if product_id is not None and product_id in actual:
            hits += 1
            ap_sum += hits / (i + 1)
    return ap_sum / k


def normalized_average_precision(actual, recommended, k=30):
    actual = set(actual)
    if len(actual) == 0:
        return 0.0

    ap = average_precision(actual, recommended, k=k)
    ap_ideal = average_precision(actual, list(actual)[:k], k=k)
    return ap / ap_ideal


def recall_k(actual, recommended, k=30):
    return len(set(actual).intersection(set(recommended[:k]))) / max(
        len(set(actual)), 1
    )


def get_shard_path(n_shard, jsons_dir=cfg.JSONS_DIR):
    return "{}/{:02d}.jsons.splitted".format(jsons_dir, n_shard)


def md5_hash(x):
    return int(hashlib.md5(x.encode()).hexdigest(), 16)


def get_check_users():
    check_users = []
    with open(cfg.CHECK_QUERY_PATH) as f:
        for line in f:
            query_data, _ = line.strip().split("\t")
            client_id = json.loads(query_data)["client_id"]
            check_users.append(client_id)
    return check_users
