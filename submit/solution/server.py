import sys
sys.path.insert(0, "src")

import datetime as dt
from flask import Flask, jsonify, request
from predictors import GBMPredictor
from pathlib import Path

app = Flask(__name__)

ASSETS_DIR = Path("assets")
PREDICTOR = GBMPredictor(
    lgbm_model_path=str(ASSETS_DIR / "lgbm_model.txt"),
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
    N=100,
    trunk_svd_arr_path=ASSETS_DIR / "svd_128_components_T.npy",
    faiss_index_path=str(ASSETS_DIR / "faiss_base.idx"),
    train_scores_path=ASSETS_DIR / "X_scores_sparse.npz",
    faiss_neighbors=512,
    faiss_nprobe=16,
)

@app.route("/ready")
def ready():
    return "OK"


@app.route("/recommend", methods=["POST"])
def recommend():
    r = request.json

    result = PREDICTOR.predict(r, PREDICTOR.lgb_model)
    return jsonify({"recommended_products": result})


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=8000)