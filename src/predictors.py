import random
import datetime as dt
import itertools as it
import pickle
from collections import defaultdict, Counter

import faiss
import lightgbm as lgb
import math
import numpy as np
import pandas as pd
import feather
from catboost import CatBoost
from scipy import sparse as sp

from utils import ProductEncoder, ProductFeatEncoder, make_coo_row


class GBMFeatures:
    def __init__(
        self,
        product_csv_path,
        model_pickled_path,
        products_misc_path,
        implicit_tfidf_path=None,
        product_features_encoder_path=None,
        implicit_als_path=None,
        N=100,
        cache_fm_path=None,
        implicit_cosine2_path=None,
        fm_features_feather_path=None,
        trunk_svd_arr_path=None,
        faiss_index_path=None,
        train_scores_path=None,
        umap_item_emb_path=None,
        faiss_neighbors=128,
        faiss_nprobe=32,
        item_co_occurrence_path=None,
        item_occurrence_path=None,
        user_prod_log_idf_path=None,
        tran_prod_log_idf_path=None,
        random_seed=0,
    ):
        self.product_encoder = ProductEncoder(product_csv_path)

        if not product_features_encoder_path:
            self.product_features = ProductFeatEncoder(product_csv_path)
        else:
            self.product_features = pickle.load(
                open(product_features_encoder_path, "rb")
            )

        self.model = pickle.load(open(model_pickled_path, "rb"))
        self.init_misc_features(products_misc_path, N)

        if implicit_tfidf_path:
            self.implicit_tfidf = pickle.load(open(implicit_tfidf_path, "rb"))
        if implicit_als_path:
            self.implicit_als = pickle.load(open(implicit_als_path, "rb"))
        if implicit_cosine2_path:
            self.implicit_cosine2 = pickle.load(open(implicit_cosine2_path, "rb"))

        if cache_fm_path is not None:
            self.cache_fm_feat = pickle.load(open(cache_fm_path, "rb"))
        else:
            self.cache_fm_feat = defaultdict(dict)

        if trunk_svd_arr_path:
            self.trunk_svd_arr = np.load(trunk_svd_arr_path)
        if train_scores_path:
            self.X_scores = sp.load_npz(train_scores_path)

        self.faiss_neighbors = faiss_neighbors
        self.faiss_nprobe = faiss_nprobe
        if faiss_index_path:
            self.faiss_index = faiss.read_index(faiss_index_path)
            self.faiss_index.nprobe = faiss_nprobe

        if item_occurrence_path:
            # add 1 to be able to divide co_occurrence features on occurrence
            self.item_occurrence = np.load(item_occurrence_path) + 1
        if item_co_occurrence_path:
            self.item_co_occurrence = sp.load_npz(item_co_occurrence_path)

        if user_prod_log_idf_path:
            self.user_prod_log_idf = np.load(user_prod_log_idf_path)
        if tran_prod_log_idf_path:
            self.tran_prod_log_idf = np.load(tran_prod_log_idf_path)

        if umap_item_emb_path:
            self.umap_item_emb = np.load(umap_item_emb_path)

        if fm_features_feather_path:
            self.feather_fm_features = feather.read_dataframe(fm_features_feather_path)
            self.feather_names = self.feather_fm_features.columns
            self.feather_values = self.feather_fm_features.values
            self.get_implicit_features = self.get_feather_features
        else:
            self.get_implicit_features = self.get_implicit_train_model_features

        self.feature_extractor = {"gender": {"M": 0, "F": 1, "U": 2}}

        self.N = N
        self.N_05 = self.N // 2
        self.pos_0_score = 1000
        self.datetime_stamp_2000 = dt.datetime(2000, 1, 1).toordinal()

        # cache
        self.default_history_feat = defaultdict(dict)
        self.date_cache = {}
        self.cooccur_default_values = {}

        random.seed(random_seed)

    def init_misc_features(self, path, N):
        self.df_products_misc = pd.read_csv(path, index_col=0)
        self.pos_in_top = self.df_products_misc["popularity_position"].to_dict()
        self.top_products = {pos: idx for idx, pos in self.pos_in_top.items()}
        self.product_cost = self.df_products_misc["cost"].round(2).to_dict()
        self.top_n_items = list(self.top_products.values())[:N]

    # TODO: refactor
    def get_items_pool(self, product_ind_csr, product_ind_csr_not_normed):
        selected_items = set()

        # add cosine 50 items
        recs_cosine50 = self.model.recommend(
            userid=0,
            user_items=product_ind_csr,
            N=self.N,
            filter_already_liked_items=False,
            recalculate_user=True,
        )
        selected_items |= set(item[0] for item in recs_cosine50)

        # add tfidf items
        recs_tfidf = self.implicit_tfidf.recommend(
            userid=0,
            user_items=product_ind_csr,
            N=self.N,
            filter_already_liked_items=False,
            recalculate_user=True,
        )
        selected_items |= set(item[0] for item in recs_tfidf)

        # add cosine2 items
        recs_cosine2 = self.implicit_cosine2.recommend(
            userid=0,
            user_items=product_ind_csr,
            N=self.N,
            filter_already_liked_items=False,
            recalculate_user=True,
        )
        selected_items |= set(item[0] for item in recs_cosine2)

        # TODO: increase speed or delete
        # add ALS items
        # recs_als = self.implicit_als.recommend(
        #     userid=0, user_items=product_ind_csr_not_normed, N=self.N,
        #     filter_already_liked_items=False, recalculate_user=True,
        # )
        # selected_items |= set(item[0] for item in recs_als)

        # add top items
        selected_items |= set(self.top_n_items)

        # add top history items
        if product_ind_csr.nnz > 0:
            selected_items |= set(
                self._get_top_recommendations(
                    product_ind_csr.A[0], k=min(product_ind_csr.nnz + 1, self.N_05)
                )[0]
            )
        return selected_items

    def get_default_history_feat(self, idx, cache=True):
        if cache and idx in self.default_history_feat:
            return self.default_history_feat[idx]

        product_feats = {}
        product_feats["quantity"] = 0
        product_feats["items_in_different_transactions"] = 0
        product_feats["mean_items_in_trans"] = 0
        product_feats["mean_items_in_trans_log"] = 0
        product_feats["mean_items_in_history"] = 0
        product_feats["mean_items_in_trans_log_denom"] = 0

        product_feats["user_item_idf"] = -1
        product_feats["user_tran_idf_mult_total_idf"] = -1

        product_feats["total_items_in_history_pct"] = 0
        product_feats["item_spent"] = 0
        product_feats["item_cost"] = self.product_cost[idx]

        product_feats["item_is_in_last_transaction"] = 0
        product_feats["last_item_purchase_num_days"] = -1
        product_feats["mean_item_days_between_purchase"] = -1

        product_feats["cooc_item_sum_score"] = -1
        product_feats["cooc_item_mean_score"] = -1
        product_feats["cooc_mean_from_items"] = -1
        product_feats["cooc_sum_from_items"] = -1
        product_feats["cooc_max_from_items"] = -1

        if cache:
            self.default_history_feat[idx] = product_feats
        return self.default_history_feat[idx]

    def to_date(self, x, cache=True):
        """
        Convert datetime to datetime.date

        # format for training and inference is different:
        # training: "%Y-%m-%d %H:%M:%S"
        # inference: "%Y-%m-%dT%H:%M:%S"
        """
        str_date = x[:10]
        if str_date in self.date_cache:
            return self.date_cache[str_date]

        # self.date_cache[str_date] = dt.datetime.strptime(str_date, "%Y-%m-%d")
        # faster approach
        self.date_cache[str_date] = dt.datetime(*map(int, str_date.split("-")))
        return self.date_cache[str_date]

    def get_product_feat_from_history(self, transaction_history, cooc_scores, item2pos):
        product_feats = defaultdict(Counter)
        if not transaction_history:
            return product_feats

        items_cooc_scores = np.zeros(len(cooc_scores))
        items_dates_purchase = defaultdict(list)
        num_items_in_purchase = 0

        for txn in transaction_history:
            purchase_date = self.to_date(txn["datetime"])
            product_ind = self.product_encoder.toIdx(
                item["product_id"] for item in txn["products"]
            )
            product_cooc_pos = [item2pos[pidx] for pidx in product_ind]
            items_cooc_scores[product_cooc_pos] += cooc_scores[product_cooc_pos].sum(
                axis=1
            )

            for pidx, item in zip(product_ind, txn["products"]):
                product_feats[pidx]["quantity"] += item["quantity"]
                product_feats[pidx]["items_in_different_transactions"] += 1
                num_items_in_purchase += 1
                items_dates_purchase[pidx].append(purchase_date)

        num_items_in_purchase = max(num_items_in_purchase, 1)
        len_trans = max(len(transaction_history), 1)
        cooc_mean_from_items = items_cooc_scores.mean()
        cooc_sum_from_items = items_cooc_scores.sum()
        cooc_max_from_items = items_cooc_scores.max()

        for pidx, item in product_feats.items():
            num_trans_with_item = item["items_in_different_transactions"]
            item["mean_items_in_trans"] = num_trans_with_item / len_trans
            # 1e-2 for better distribution
            item["mean_items_in_trans_log"] = np.log(
                num_trans_with_item / len_trans + 1e-2
            )
            item["mean_items_in_history"] = num_trans_with_item / num_items_in_purchase
            item["mean_items_in_trans_log_denom"] = num_trans_with_item / (
                1 + math.log(len_trans)
            )

            # iDF scores, usual item-item TF-iDF is calculated via the implicit model
            item["user_item_idf"] = self.user_prod_log_idf[pidx]
            item["user_tran_idf_mult_total_idf"] = (
                math.log(len_trans / num_trans_with_item) * self.tran_prod_log_idf[pidx]
            )

            max_quantity = max(item["quantity"], 1)
            item["total_items_in_history_pct"] = max_quantity / num_items_in_purchase
            item["item_spent"] = self.product_cost[pidx] * max_quantity
            item["item_cost"] = self.product_cost[pidx]

            item["item_is_in_last_transaction"] = int(
                items_dates_purchase[pidx][-1] == purchase_date
            )
            item["last_item_purchase_num_days"] = (
                purchase_date - items_dates_purchase[pidx][-1]
            ).days

            item["cooc_item_sum_score"] = items_cooc_scores[item2pos[pidx]]
            item["cooc_item_mean_score"] = (
                items_cooc_scores[item2pos[pidx]] / num_trans_with_item
            )
            item["cooc_mean_from_items"] = cooc_mean_from_items
            item["cooc_sum_from_items"] = cooc_sum_from_items
            item["cooc_max_from_items"] = cooc_max_from_items

            if len(items_dates_purchase[pidx]) > 1:
                items_purchase = items_dates_purchase[pidx]
                items_days_between_purchase = [
                    (end_date - start_date).days
                    for start_date, end_date in zip(
                        items_purchase[:-1], items_purchase[1:]
                    )
                ]
                item["mean_item_days_between_purchase"] = sum(
                    items_days_between_purchase
                ) / len(items_days_between_purchase)
            else:
                item["mean_item_days_between_purchase"] = -1

        return product_feats

    def get_highlevel_feat_from_history(self, transaction_history):
        feats = {}
        num_items_in_purchase = []
        uniq_stores = set()
        purchase_sum = 0
        prev_date = None
        dates_intervals = []
        min_date = None

        for txn in transaction_history:
            purchase_sum += txn["purchase_sum"]
            uniq_stores.add(txn["store_id"])
            num_items_in_purchase.append(len(txn["products"]))

            cur_date = self.to_date(txn["datetime"])
            if prev_date:
                date_diff = (cur_date - prev_date).days
                dates_intervals.append(date_diff)

            if min_date is None:
                min_date = cur_date

            prev_date = cur_date

        total_purchases = sum(num_items_in_purchase)
        feats["purchase_sum"] = purchase_sum
        feats["num_uniq_store"] = len(uniq_stores)
        feats["mean_num_items"] = total_purchases / max(len(num_items_in_purchase), 1)
        feats["total_bought_items"] = total_purchases
        feats["max_bought_items"] = (
            max(num_items_in_purchase) if num_items_in_purchase else 0
        )
        feats["mean_days_between_purchases"] = len(dates_intervals) / max(
            sum(dates_intervals), 1
        )
        feats["max_days_between_purchases"] = (
            max(dates_intervals) if dates_intervals else -1
        )

        if min_date is not None:
            feats["days_between_first_and_last_purchase"] = (cur_date - min_date).days
        else:
            feats["days_between_first_and_last_purchase"] = -1

        return feats

    def get_implicit_model_features(
        self,
        model,
        idx,
        model_prefix="fm",
        n=5,
        k_score=0,
        skip_first=False,
        only_score=True,
        cache=True,
        only_second=True,
    ):
        model_cache_name = str(model.__class__) + model_prefix
        if cache and idx in self.cache_fm_feat[model_cache_name]:
            return self.cache_fm_feat[model_cache_name][idx]

        similar_items = model.similar_items(idx, max(k_score, n))
        fm_features = {}
        for i, (_, score) in enumerate(similar_items[:n]):
            if i == 0 and skip_first:
                continue

            fm_features[f"top_item_{model_prefix}_{i}_score"] = score

        if k_score != 0:
            fm_features[f"sum_top_{model_prefix}_{k_score}_score"] = sum(
                score for _, score in similar_items
            )
            fm_features[f"mean_top_{model_prefix}_{k_score}_score"] = np.mean(
                [score for _, score in similar_items]
            )

        if cache:
            self.cache_fm_feat[model_cache_name][idx] = fm_features
        return fm_features

    def get_feather_features(self, idx):
        """Transform feather dataframe to features dict."""
        scores = self.feather_values[idx]
        return {key: val for key, val in zip(self.feather_names, scores)}

    def get_implicit_train_model_features(self, idx):
        return {
            **self.get_implicit_model_features(
                self.model, idx, "cosine50", n=3, k_score=0
            ),
            **self.get_implicit_model_features(
                self.implicit_cosine2, idx, "cosine2", n=2, skip_first=True, k_score=0
            ),
            **self.get_implicit_model_features(
                self.implicit_tfidf, idx, "tfidf50", n=3, skip_first=True, k_score=0
            ),
            **self.get_implicit_model_features(
                self.implicit_als, idx, "als", n=3, k_score=0
            ),
        }

    @staticmethod
    def _get_top_recommendations(row, k=100):
        k = min(len(row), k)
        ind = np.argpartition(row, -k)[-k:]
        top_k_ind = ind[np.argsort(row[ind])][::-1]
        return top_k_ind, row[top_k_ind]

    def _get_faiss_scores(self, product_ind_csr):
        x_dense = product_ind_csr * self.trunk_svd_arr
        faiss_result = self.faiss_index.search(x_dense, self.faiss_neighbors)
        neighbors = faiss_result[1]
        scores = np.asarray(faiss_result[0] * self.X_scores[neighbors[0]]).flatten()
        return scores, faiss_result

    def get_faiss_features(self, product_ind_csr, selected_items):
        scores, faiss_result = self._get_faiss_scores(product_ind_csr)
        top_k_ind, sorted_predictions = self._get_top_recommendations(
            scores[selected_items], k=10000
        )

        top_neighbor = faiss_result[0][0][0]
        faiss_neighbor_mean = faiss_result[0].mean()
        non_zero_idx = scores.nonzero()[0]
        non_zero_scores = len(non_zero_idx)
        sum_scores = scores.sum()
        features = defaultdict(dict)

        pos = 0
        prev_score = -10
        for idx, score in zip(top_k_ind, sorted_predictions):
            if score == 0:
                pos = self.pos_0_score

            pidx = selected_items[idx]
            features[pidx]["faiss_score"] = score
            features[pidx]["faiss_pos"] = pos
            features[pidx]["faiss_neighbor_top_score"] = top_neighbor
            features[pidx]["faiss_neighbor_mean_score"] = faiss_neighbor_mean
            features[pidx]["faiss_num_non_zero_scores"] = non_zero_scores
            features[pidx]["faiss_scores_sum"] = sum_scores

            if prev_score != score:
                pos += 1
            prev_score = score
        return features

    def get_umap_scores(self, product_ind_csr, selected_items):
        product_emb = (product_ind_csr * self.umap_item_emb)[0]

        features = defaultdict(dict)
        for pidx in selected_items:
            features[pidx]["umap_user_emb_0"] = product_emb[0]
            features[pidx]["umap_user_emb_1"] = product_emb[1]

            item_emb = self.umap_item_emb[pidx]
            features[pidx]["umap_item_emb_0"] = item_emb[0]
            features[pidx]["umap_item_emb_1"] = item_emb[1]

            user_item_emb = np.mean([product_emb, item_emb], axis=0)
            features[pidx]["umap_user_item_emb_0"] = user_item_emb[0]
            features[pidx]["umap_user_item_emb_1"] = user_item_emb[1]

        return features

    @staticmethod
    def _implicit_rank(model, product_ind_csr, selected_items):
        predictions = model.rank_items(
            0, product_ind_csr, selected_items, recalculate_user=True
        )
        return predictions

    @staticmethod
    def _custom_implicit_rank(model, product_ind_csr, selected_items):
        recommendations = product_ind_csr.dot(model.similarity)
        predictions = sorted(
            zip(selected_items, recommendations[0, selected_items].A[0]),
            key=lambda x: -x[1],
        )
        return predictions

    def get_implicit_scores(
        self, model, product_ind_csr, selected_items, model_prefix="tfidf"
    ):
        if model_prefix in ("tfidf", "cosine2", "cosine50"):
            sorted_predictions = self._custom_implicit_rank(
                model, product_ind_csr, selected_items
            )
        else:
            sorted_predictions = self._implicit_rank(
                model, product_ind_csr, selected_items
            )

        features = defaultdict(dict)
        pos = 0
        prev_score = -10
        for pidx, score in sorted_predictions:
            if score == 0:
                pos = self.pos_0_score

            features[pidx][f"{model_prefix}_score"] = score
            features[pidx][f"{model_prefix}_pos"] = pos

            if prev_score != score:
                pos += 1
            prev_score = score
        return features

    def get_num_days_from_last_transaction(self, js, last_transaction_date):
        if last_transaction_date is None:
            return -1

        try:
            query_date = self.to_date(js.get("query_time"))
            num_days_from_last_transaction = (query_date - last_transaction_date).days

        except TypeError:
            target_date = self.to_date(js["target"][0]["datetime"])
            num_days_from_last_transaction = (target_date - last_transaction_date).days
            num_days_from_last_transaction = random.randint(
                0, num_days_from_last_transaction
            )

        return num_days_from_last_transaction

    def get_co_occurrence_features(
        self,
        cooc_scores,
        weights=None,
        prefix="co_occurrence",
        default_value=-1,
        aggs=("max", "mean", "sum"),
    ):
        keys = [f"{prefix}_{agg_name}" for agg_name in aggs]

        if weights is not None:
            keys += [f"{key}_weighted" for key in keys]

        if cooc_scores.size == 0:
            if prefix not in self.cooccur_default_values:
                self.cooccur_default_values[prefix] = dict(
                    zip(keys, it.repeat(default_value))
                )
            return it.repeat(
                self.cooccur_default_values[prefix], 1000
            )  # to prevent endless iterations

        scores = []
        keys = []
        for agg_name in aggs:
            agg_func = getattr(np, agg_name)
            scores.append(agg_func(cooc_scores, axis=1))
            keys.append(f"{prefix}_{agg_name}")

        if weights is not None:
            cooc_scores_w = cooc_scores * weights
            for agg_name in aggs:
                agg_func = getattr(np, agg_name)
                scores.append(agg_func(cooc_scores_w, axis=1))
                keys.append(f"{prefix}_{agg_name}_weighted")

        features = []
        for values in np.vstack(scores).T:
            features.append(dict(zip(keys, values)))
        return iter(features)

    def get_cooc_features(
        self, cooc_purchased_all_scores, selected_items, purchased_items, cooc_weights
    ):
        cooc_scores = cooc_purchased_all_scores[selected_items].A

        cooc_scores_norm_item = (
            cooc_scores / self.item_occurrence[selected_items][:, None]
        )
        cooc_scores_norm_co_item = cooc_scores / self.item_occurrence[purchased_items]

        cooc_norm_item_features = self.get_co_occurrence_features(
            cooc_scores_norm_item, cooc_weights, "co_occurrence_item_norm"
        )
        cooc_scores_norm_co_item_features = self.get_co_occurrence_features(
            cooc_scores_norm_co_item, cooc_weights, "co_occurrence_co_item_norm"
        )
        return cooc_norm_item_features, cooc_scores_norm_co_item_features

    def get_gbm_features(
        self, js, train=False, drop_null_target_records=False, add_target_records=False
    ):
        # sort history as in public and check it was unordered
        js["transaction_history"] = sorted(
            js["transaction_history"], key=lambda x: x["datetime"]
        )

        if train:
            target_products = set(
                self.product_encoder.toIdx(
                    [pid for pid in js["target"][0]["product_ids"]]
                )
            )

        transaction_history = js.get("transaction_history", [])
        if transaction_history:
            last_transaction_date = self.to_date(
                transaction_history[-1].get("datetime")
            )
            # num days from 2000/1/1
            last_transaction_timestamp = (
                last_transaction_date.toordinal() - self.datetime_stamp_2000
            )
        else:
            last_transaction_date = None
            last_transaction_timestamp = None
        num_days_from_last_transaction = self.get_num_days_from_last_transaction(
            js, last_transaction_date
        )

        product_ind_csr_not_normed = make_coo_row(
            transaction_history, self.product_encoder, normalize=False
        ).tocsr()
        product_ind_csr = make_coo_row(
            js.get("transaction_history", []), self.product_encoder
        ).tocsr()

        selected_items = list(
            self.get_items_pool(product_ind_csr, product_ind_csr_not_normed)
        )

        cosine50_scores = self.get_implicit_scores(
            self.model, product_ind_csr, selected_items, model_prefix="cosine50"
        )
        tf_idf_scores = self.get_implicit_scores(
            self.implicit_tfidf, product_ind_csr, selected_items, model_prefix="tfidf"
        )
        cosine2_scores = self.get_implicit_scores(
            self.implicit_cosine2,
            product_ind_csr,
            selected_items,
            model_prefix="cosine2",
        )
        als_scores = self.get_implicit_scores(
            self.implicit_als,
            product_ind_csr_not_normed,
            selected_items,
            model_prefix="als",
        )

        # co occurrence features
        # all user items purchases
        purchased_items = product_ind_csr.indices
        cooc_weights = product_ind_csr.data
        cooc_purchased_all_scores = self.item_co_occurrence[:, purchased_items]

        cooc_norm_item_features, cooc_scores_norm_co_item_features = self.get_cooc_features(
            cooc_purchased_all_scores, selected_items, purchased_items, cooc_weights
        )

        # scores per each transaction
        cooc_purchased_scores = cooc_purchased_all_scores[purchased_items].A
        cooc_purchased_scores_norm_co_item = (
            cooc_purchased_scores / self.item_occurrence[purchased_items]
        )
        purchased_item2pos = {pid: pos for pos, pid in enumerate(purchased_items)}

        product_history_features = self.get_product_feat_from_history(
            js.get("transaction_history", []),
            cooc_purchased_scores_norm_co_item,
            purchased_item2pos,
        )

        high_level_features = self.get_highlevel_feat_from_history(
            js.get("transaction_history", [])
        )

        # faiss_features = self.get_faiss_features(product_ind_csr, selected_items)
        umap_scores = self.get_umap_scores(product_ind_csr, selected_items)

        gbm_records = []
        for product_idx in selected_items:
            record = dict(
                **{
                    "idx": product_idx,
                    "age": js["age"],
                    "gender": self.feature_extractor["gender"][js["gender"]],
                    "num_transactions": len(js.get("transaction_history", [])),
                    "popularity_position": self.pos_in_top[product_idx],
                    "last_transaction_timestamp": last_transaction_timestamp,
                    "num_days_from_last_transaction": num_days_from_last_transaction,
                },
                **high_level_features,
                **product_history_features.get(
                    product_idx, self.get_default_history_feat(product_idx)
                ),
                **self.product_features.product_features(
                    self.product_encoder.toPid(int(product_idx))
                ),
                **self.get_implicit_features(product_idx),
                **als_scores[product_idx],
                **tf_idf_scores[product_idx],
                **cosine50_scores[product_idx],
                **cosine2_scores[product_idx],
                **next(cooc_norm_item_features),
                **next(cooc_scores_norm_co_item_features),
                # **faiss_features[product_idx],
                **umap_scores[product_idx],
            )

            record["item_pct_spent"] = record.get("item_spent", 0) / max(
                record.get("purchase_sum", 1), 1
            )

            if train:
                record["target"] = int(product_idx in target_products)
                record["client_id"] = js["client_id"]

            gbm_records.append(record)

        if train:
            gt_products = dict(
                client_id=js["client_id"], products=list(target_products)
            )
            return gbm_records, gt_products

        return gbm_records


class GBMPredictor(GBMFeatures):
    def __init__(
        self,
        product_csv_path,
        *args,
        lgbm_model_path=None,
        cat_model_path=None,
        **kwargs,
    ):
        super(GBMPredictor, self).__init__(product_csv_path, *args, **kwargs)
        self.product_encoder = ProductEncoder(product_csv_path)

        if lgbm_model_path:
            self.lgb_model = lgb.Booster(model_file=lgbm_model_path)

        if cat_model_path:
            self.cat_model = CatBoost().load_model(cat_model_path)

    @staticmethod
    def predict_proba(X, model):
        pred = model.predict(X)
        return pred

    def sort_predictions(self, product_idx, gbm_pred, n=30):
        product_idx_sorted, _ = zip(
            *sorted(zip(product_idx, gbm_pred), key=lambda x: -x[1])
        )
        product_ids = self.product_encoder.toPid(
            [idx for idx in product_idx_sorted[:n]]
        )
        return product_ids

    # TODO: add different blending weights
    def predict(self, js, models):
        X = self.get_gbm_features(js)
        feature_values = [list(item.values()) for item in X]
        product_idx = [item["idx"] for item in X]
        if isinstance(models, (list, tuple)):
            gbm_pred = np.zeros_like(product_idx, dtype=float)
            for model in models:
                gbm_pred += 0.5 * self.predict_proba(feature_values, model)
        else:
            gbm_pred = self.predict_proba(feature_values, models)

        return self.sort_predictions(product_idx, gbm_pred)
