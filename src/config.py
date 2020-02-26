# all path are from ./src folder
from pathlib import Path

PURCHASE_CSV_PATH = "../data/raw/purchases.csv"
CLIENT_CSV_PATH = "../data/raw/clients.csv"
PRODUCT_CSV_PATH = "../data/raw/products.csv"
CHECK_QUERY_PATH = "../data/raw/check_queries.tsv"
JSONS_DIR = "../tmp/jsons/"
MAX_CHUNKS = None
NUM_SHARDS = 16
ASSETS_DIR = Path("../submit/solution/assets")

# determed from check quieries
BASE_SPLIT_POINT = "2019-03-02 10:05:00"
