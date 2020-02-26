import sys
import json
import requests
import datetime as dt

import numpy as np
import tqdm 


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


def run_queries(url, queryset_file, max_queries=1000):
    ap_values = []
    durations = []

    total_records = 0
    with open(queryset_file) as fin:
        for line in fin:
            total_records += 1

    total_records = min(total_records, max_queries)
    
    with open(queryset_file) as fin:
        for i, line in enumerate(tqdm.tqdm(fin, total=total_records)):
            splitted = line.strip().split('\t')
            if len(splitted) == 1:
                query_data = json.loads(splitted[0])
                next_transaction = query_data['target'][0]
            else:
                query_data, next_transaction = map(json.loads, splitted)
            
            start_time = dt.datetime.now()
            # resp = requests.post(url, json=query_data, timeout=0.3)
            resp = requests.post(url, json=query_data)
            duration = (dt.datetime.now() - start_time).total_seconds()
            durations.append(duration)
            resp.raise_for_status()
            resp_data = resp.json()
            
            if len(set(resp_data['recommended_products'])) < 30:
                print(query_data)
                print(resp_data)

            assert len(resp_data['recommended_products']) == 30
            assert len(set(resp_data['recommended_products'])) == 30
            assert all(isinstance(item, str) for item in resp_data['recommended_products'])
            assert "recommended_products" in resp_data
            
            ap = normalized_average_precision(next_transaction['product_ids'], resp_data['recommended_products'])
            ap_values.append(ap)
            
            if i >= max_queries:
                break
            
    map_score = sum(ap_values) / len(ap_values)
    print("Max time:", np.max(durations), "mean_time:", np.mean(durations), "min_time:", np.min(durations))
    return map_score


if __name__ == '__main__':
    url = sys.argv[1] # 'http://localhost:8000/recommend'
    queryset_file = sys.argv[2] # 'data/check_queries.tsv'
    max_queries = int(sys.argv[3]) # 1000
    score = run_queries(url, queryset_file, max_queries)
    print('Score:', score)
