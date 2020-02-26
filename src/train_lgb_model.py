import argparse
import gc
import json
import logging
import pprint
import sys
from pathlib import Path

import feather
import numpy as np
import lightgbm as lgb
import pandas as pd
from scipy import sparse as sp
from tqdm import tqdm

import config as cfg
from predictors import GBMFeatures, GBMPredictor
from utils import (
    ProductEncoder,
    make_coo_row,
    normalized_average_precision,
    get_shard_path,
    cache_to_feather,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("lgb_model.log"), logging.StreamHandler()],
)


def evalute_queries(queryset_file, max_records=1000):
    check_scores = []
    with open(queryset_file) as fin:
        for i, line in enumerate(tqdm(fin)):
            splitted = line.strip().split("\t")
            if len(splitted) == 1:
                query_data = json.loads(splitted[0])
                next_transaction = query_data["target"][0]
            else:
                query_data, next_transaction = map(json.loads, splitted)
                query_data["target"] = [next_transaction]

            query_data["transaction_history"] = sorted(
                query_data["transaction_history"], key=lambda x: x["datetime"]
            )
            recommended_items = PREDICTOR.predict(query_data, PREDICTOR.lgb_model)

            gt_items = query_data["target"][0]["product_ids"]
            nap = normalized_average_precision(gt_items, recommended_items)
            check_scores.append(nap)

            if i == max_records:
                break
    return np.mean(check_scores)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    N_POOL = args.N

    ASSETS_DIR = cfg.ASSETS_DIR
    SHARDS = range(cfg.NUM_SHARDS - 1)
    NUM_TEST_SHARD = cfg.NUM_SHARDS - 1

    TRAIN_DIR = Path(f"../tmp/train_chunks_{N_POOL}")
    TEST_DIR = Path(f"../tmp/test_chunks_{N_POOL}")
    logger = logging.getLogger(__name__)
    product_encoder = ProductEncoder(cfg.PRODUCT_CSV_PATH)

    logger.info("Loading train dataset")
    dfs = []
    for num_shard in tqdm(SHARDS, leave=False):
        if Path(f"{TRAIN_DIR}/df_train_{num_shard}.feather").exists():
            dfs.append(
                feather.read_dataframe(f"{TRAIN_DIR}/df_train_{num_shard}.feather")
            )

    logger.info("Join chunks to full train dataframe")
    df_gbm_train = pd.concat(dfs, sort=False)
    logger.info(f"Shape of the train dataframe {df_gbm_train.shape}")

    del dfs
    gc.collect()

    logger.info("Loading test dataset")
    df_gbm_test = feather.read_dataframe(TEST_DIR / f"df_test_{NUM_TEST_SHARD}.feather")
    gt_all_rec_test = []
    for js in tqdm(
        (json.loads(s) for s in open(get_shard_path(NUM_TEST_SHARD))), leave=False
    ):
        target_products = set(
            product_encoder.toIdx([pid for pid in js["target"][0]["product_ids"]])
        )
        gt_products = dict(client_id=js["client_id"], products=list(target_products))
        gt_all_rec_test.append(gt_products)
    logger.info(f"Shape of the test dataframe {df_gbm_test.shape}")

    logger.info("Add query_id column")
    df_gbm_train["query_id"] = df_gbm_train.groupby("client_id").ngroup()
    df_gbm_test["query_id"] = df_gbm_test.groupby("client_id").ngroup()

    logger.info("Build LGB datasets")
    drop_cols = ["client_id", "target", "query_id"]
    train_ds = lgb.Dataset(
        df_gbm_train.drop(drop_cols, errors="ignore", axis=1),
        df_gbm_train["target"],
        group=df_gbm_train["query_id"].value_counts().sort_index().values,
    )
    test_ds = lgb.Dataset(
        df_gbm_test.drop(drop_cols, errors="ignore", axis=1),
        df_gbm_test["target"],
        group=df_gbm_test["query_id"].value_counts().sort_index().values,
    )

    lgb_params = dict(
        objective="binary",
        #     objective='lambdarank',
        max_depth=12,
        random_state=42,
        learning_rate=0.05,
        lambda_l2=10,
        metric=("binary", "map"),
        eval_at=30,
        max_bin=63,
        first_metric_only=True,
    )
    num_boost_round = 6000
    logger.info("LGB params:\n%s", pprint.pformat(lgb_params))

    gbm = lgb.train(
        lgb_params,
        train_ds,
        num_boost_round,
        valid_sets=(train_ds, test_ds),
        verbose_eval=10,
        early_stopping_rounds=100,
    )

    drop_cols = ["client_id", "target", "lgb_scores", "query_id"]
    lgb_scores = gbm.predict(df_gbm_test.drop(drop_cols, axis=1, errors="ignore"))
    df_gbm_test["lgb_scores"] = lgb_scores

    lgb_ranked = (
        df_gbm_test.groupby("client_id")[["idx", "lgb_scores"]]
        .apply(
            lambda x: x.sort_values("lgb_scores", ascending=False)[:30]["idx"].tolist()
        )
        .to_dict()
    )

    gt_test = {item["client_id"]: item["products"] for item in gt_all_rec_test}
    scores = []
    for client_id, recommended_idx in lgb_ranked.items():
        ap = normalized_average_precision(gt_test[client_id], recommended_idx)
        scores.append(ap)
    model_score = np.mean(scores)
    logger.info(f"Test score: {model_score}")

    params_str = "__".join(
        "_".join(map(str, item)) for item in gbm.params.items() if item[0] != "metric"
    )
    model_filename = f"lgbm_model__pool_{N_POOL}__{params_str}__{model_score:.6f}.txt"
    model_path = str(ASSETS_DIR / model_filename)
    gbm.save_model(model_path)
    logger.info(f"Model was saved to {model_path}")

    # Check predictor
    PREDICTOR = GBMPredictor(
        lgbm_model_path=str(ASSETS_DIR / model_filename),
        product_csv_path=ASSETS_DIR / "products.csv",
        model_pickled_path=ASSETS_DIR / "model_implicit_cosine_50.pkl",
        products_misc_path=ASSETS_DIR / "products_misc.csv",
        product_features_encoder_path=ASSETS_DIR / "product_features.pkl",
        implicit_tfidf_path=ASSETS_DIR / "model_implicit_tf_idf100.pkl",
        implicit_als_path=ASSETS_DIR / "model_implicit_als_16fact_12iter.pkl",
        fm_features_feather_path=ASSETS_DIR / "implicit_scores.feather",
        implicit_cosine2_path=ASSETS_DIR / "model_implicit_cosine2.pkl",
        umap_item_emb_path=ASSETS_DIR / "umap_item_emb.npy",
        item_co_occurrence_path=ASSETS_DIR / "item_co_occurrence_min_cnt_5.npz",
        item_occurrence_path=ASSETS_DIR / "item_occurrence.npy",
        user_prod_log_idf_path=ASSETS_DIR / "user_prod_log_idf.npy",
        tran_prod_log_idf_path=ASSETS_DIR / "tran_prod_log_idf.npy",
        N=N_POOL,
        # trunk_svd_arr_path=ASSETS_DIR / "svd_128_components_T.npy",
        # faiss_index_path=str(ASSETS_DIR / "faiss_base.idx"),
        # train_scores_path=ASSETS_DIR / "X_scores_sparse.npz",
        # faiss_neighbors=512,
        # faiss_nprobe=8,
    )

    # check queries
    check_queryset_file = cfg.CHECK_QUERY_PATH
    logger.info(f"Evaluating check queries {check_queryset_file}")
    check_score = evalute_queries(check_queryset_file)
    logger.info(f"Check score: {check_score}")

    # test queries
    max_records = 1000
    queryset_file = f"{cfg.JSONS_DIR}/{NUM_TEST_SHARD}.jsons.splitted"
    logger.info(
        f"Evaluating test queries {queryset_file} with {max_records} max_records"
    )
    test_score = evalute_queries(queryset_file, max_records=max_records)
    logger.info(f"Test score: {test_score}")
