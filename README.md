# 1st place solution [RetailHero.ai/#2](https://retailhero.ai/c/recommender_system/overview)

## Overview

[solution presentation](https://github.com/aprotopopov/retailhero_recommender/tree/master/slides/retailhero_recommender.pdf) (on russian)

- Data preparation, train/valid split are heavily based on excellent [baseline](https://github.com/datagym-ru/retailhero-recomender-baseline) from [@geffy](https://github.com/geffy)
- Co occurrence of purchased items per transaction
- User transactions UMAP embeddings
- Collecting miscellaneous features like `item_cost`, `popularity position`, etc
- Dataset preparation with saving each chunk as feather DataFrame
- LightGBM training on pool of items from the MF models, top products and history items

## Steps to prepare data

1. Copy data to `data/raw`

```
cd {REPO_ROOT}
mkdir -p data/raw
cp /path/to/unpacked/data/*.csv ./data/raw
cd src
```

2. Divide source purchase data into 16 shards

```bash
python purchases_to_jrows.py
```

3. Prepare train/valid data with similar structure to `check_queries.tsv`

```bash
python train_valid_split.py
```

## Train embedding and collect statistics

1. Collect co occurrence matrix

```bash
python collect_co_occur_matrix.py
```

2. Build UMAP embeddings

```bash
python train_umap_embeddings.py
```

3. Collect miscellaneous features like item popularity position, item_cost, iDF for items in transactions

```bash
python collect_miscellaneous_features.py
```

## Train models

1. Train item2item `implicit` models. It have to be launched for each model separately

```bash
python train_i2i_model.py
```

2. Collect dataset for GBM training

```bash
python collect_gbm_dataset.py
```

3. Train LGBM model

```bash
python train_lgb_model.py
```

4. Create submission file

```bash
cd submit
zip -r model.zip solution/*
```

Joint collection and training parts can be launched via `bash train.sh` as well as via notebook `sandbox/GBM.ipynb`

## The best achieved results

Scores (NMAP@30):  
Check: 0.1350  
Public: 0.1339  
Private: 0.148325  
Local: 0.155055  

[final submit](https://drive.google.com/file/d/17yR-klDIZ8vXvhTCIAEwkaBqXzuXagsg/view?usp=sharing). Code a bit messier and with less number of features but with better overall result.

*Note: LightGBM wasn't tuned carefully so higher score is achievable with the current set of features*

## Other participants solutions

- [3rd place](https://github.com/geffy/retailhero-recommender-solution) from [@geffy](https://github.com/geffy). [presentation](https://github.com/geffy/retailhero-recommender-solution/blob/master/slides/retailhero-conf.pdf)
- [8th place](https://github.com/greenwolf-nsk/retailhero-rec) from [@greenwolf-nsk](https://github.com/greenwolf-nsk)
- [9th place](https://github.com/mike-chesnokov/x5_retailhero_2020_recs) from [@mike-chesnokov](https://github.com/mike-chesnokov)
