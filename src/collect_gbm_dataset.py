import argparse
import gc
import json
import logging
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
    get_check_users,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("collect_dataset.log"), logging.StreamHandler()],
)


def get_gbm_records(shard_indices, gbm_feat, max_records=None, **kwargs):
    check_users = get_check_users()

    gbm_records = []
    num_records = 0

    for shard_idx in tqdm(shard_indices, leave=False):
        for js in tqdm(
            (json.loads(s) for s in open(get_shard_path(shard_idx))), leave=False
        ):
            if js["client_id"] in check_users:
                continue

            feat_records, _ = gbm_feat.get_gbm_features(js, train=True, **kwargs)
            gbm_records.extend(feat_records)
            num_records += 1

            if max_records and num_records >= max_records:
                return gbm_records

    return gbm_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    N_POOL = args.N
    MAX_RECORDS = args.max_records

    logger = logging.getLogger(__name__)
    ASSETS_DIR = cfg.ASSETS_DIR
    NUM_TEST_SHARD = 15

    # to test pipeline
    SHARDS = [14]

    # full training
    # SHARDS = range(15)

    gbm_feat = GBMFeatures(
        product_csv_path=ASSETS_DIR / "products.csv",
        model_pickled_path=ASSETS_DIR / "model_implicit_cosine_50.pkl",
        products_misc_path=ASSETS_DIR / "products_misc.csv",
        product_features_encoder_path=ASSETS_DIR / "product_features.pkl",
        implicit_tfidf_path=ASSETS_DIR / "model_implicit_tf_idf100.pkl",
        implicit_als_path=ASSETS_DIR / "model_implicit_als_16fact_12iter.pkl",
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
        # faiss_nprobe=16,
    )

    train_dir = Path(f"../tmp/train_chunks_{gbm_feat.N}")
    train_dir.mkdir(exist_ok=True)
    test_dir = Path(f"../tmp/test_chunks_{gbm_feat.N}")
    test_dir.mkdir(exist_ok=True)

    logger.info("Collecting train dataset")
    for num_shard in tqdm(SHARDS, leave=False):
        gbm_rec_train = get_gbm_records([num_shard], gbm_feat, max_records=MAX_RECORDS)
        df_gbm_train_chunk = pd.DataFrame(gbm_rec_train)

        train_shard_path = f"{train_dir}/df_train_{num_shard}.feather"
        logger.info(f"Saving train {num_shard} shard to {train_shard_path}")
        feather.write_dataframe(df_gbm_train_chunk, train_shard_path)

    del gbm_rec_train
    del df_gbm_train_chunk
    gc.collect()

    logger.info("Collect test dataset")
    gbm_rec_test = get_gbm_records([NUM_TEST_SHARD], gbm_feat, max_records=MAX_RECORDS)
    df_gbm_test = pd.DataFrame(gbm_rec_test)

    test_shard_path = f"{test_dir}/df_test_{num_shard}.feather"
    logger.info(f"Saving test {NUM_TEST_SHARD} shard to {test_shard_path}")
    feather.write_dataframe(df_gbm_test, test_dir / "df_test_15.feather")

    logger.info("Transform and save FM cached features to feather for faster loading")
    cache_to_feather(gbm_feat.cache_fm_feat)
